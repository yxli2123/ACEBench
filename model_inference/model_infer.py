import json
import logging
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from openai import NOT_GIVEN, OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

logging.basicConfig(level=logging.WARNING)


def convert_fc_namespace_to_dict(
    tool_calls: List[ChatCompletionMessageToolCall],
) -> List[Dict[str, Any]]:
    tool_calls_in_dict = []
    for i, tool_call in enumerate(tool_calls):
        function = tool_call.function
        arguments = function.arguments
        try:
            arguments_dict = json.loads(arguments)
        except json.JSONDecodeError as e:
            logging.warning("Skipping tool_call #%d: invalid JSON (%s)", i, e)
            continue

        tool_calls_in_dict.append(
            {"name": function.name, "arguments": arguments_dict}
        )

    return tool_calls_in_dict


class BaseModelInference:
    def __init__(
        self,
        model_name: str,
        vllm_kwargs: Optional[Dict[str, str | int]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fc_mode: bool = True,
    ):
        if vllm_kwargs is not None:
            local_model_path = vllm_kwargs["local-model-path"]
            gpu_memory_utilization = vllm_kwargs.get(
                "gpu-memory-utilization", 0.9
            )
            num_gpus = vllm_kwargs.get("num_gpus", 4)
            port = vllm_kwargs.get("port", 8181)
            model_max_len = vllm_kwargs.get("model-max-len", 40960)
            dtype = vllm_kwargs.get("dtype", "float32")

            # Example additional flags are "--enable-auto-tool-choice --tool-call-parser hermes"
            additional_args = vllm_kwargs.get("additional-args", None)

            start_vllm_cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--served-model-name",
                str(model_name).strip(),
                "--model",
                str(local_model_path),
                "--tensor-parallel-size",
                str(num_gpus).strip(),
                "--dtype",
                str(dtype),
                "--max-model-len",
                str(model_max_len).strip(),
                "--host",
                "0.0.0.0",
                "--port",
                str(port).strip(),
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
            ]

            if additional_args is not None:
                start_vllm_cmd.extend(additional_args.split())

            base_url = f"http://127.0.0.1:{port}/v1"
            api_key = "EMPTY"

            subprocess.run(start_vllm_cmd, check=True)

        self.model_name = model_name
        self.fc_mode = fc_mode

        assert all([base_url, api_key]), "base_url and api_key are required."

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: Optional[List[Any]] = NOT_GIVEN,
    ) -> ChatCompletionMessage:
        """Take the messages and return the generated text.

        messages: It has the following schema:
            [
              {"role": "system", "content": "You are a helpful assistant. You can use following tools."},
              {"role": "user", "content": "What is the weather like today?"},
              {"role": "assistant", "tool_calls": [{"type": "function", "function": "..."}]},
              {"role": "tool", "name": "get_current_weather", "content": "22.0"},
              {"role": "assistant", "content": "The weather is 22.0 degree."},
              ...
            ]
        generation_kwargs: It supports following arguments:
            {
              "temperature": 0.1,
              "max_tokens": 64,
              "top_p": 0.9,
            }
        functions: A list of functions/tools to use. If provided, evaluate the model with FC mode,
            otherwise with prompt mode.
        """
        temperature = generation_kwargs.get("temperature", 0.1)
        max_tokens = generation_kwargs.get("max_tokens", 64)
        top_p = generation_kwargs.get("top_p", 0.9)

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        message = chat_response.choices[0].message

        return message

    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: Optional[List[Any]] = NOT_GIVEN,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Implement this method.")


class UserModelInference(BaseModelInference):
    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: Optional[List[Any]] = NOT_GIVEN,
    ) -> Dict[str, Any]:
        current_message = self._generate(messages, generation_kwargs)
        current_message_text = current_message.content
        return {"role": "assistant", "content": current_message_text}


class Qwen3AgentInference(BaseModelInference):
    def __init__(
        self,
        model_name: str,
        vllm_kwargs: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fc_mode: bool = True,
    ):
        # Let the agent choose if it should call tools or return text automatically when the FC mode is on.
        if fc_mode:
            vllm_kwargs.update(
                {
                    "additional_args": "--enable-auto-tool-choice --tool-call-parser hermes"
                }
            )
        super().__init__(model_name, vllm_kwargs, base_url, api_key, fc_mode)

    @staticmethod
    def _maybe_extract_reasoning_content(text: str) -> Dict[str, str]:
        reasoning_text = ""
        if "</think>" in text:
            parts = text.split("</think>")
            reasoning_text = (
                parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            )
            text = parts[-1].lstrip("\n")

        return {"reasoning_content": reasoning_text, "response_content": text}

    @staticmethod
    def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        blobs = re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL
        )
        tools = []
        for i, b in enumerate(blobs, 1):
            try:
                obj = json.loads(b)
            except json.JSONDecodeError as e:
                logging.warning(
                    "Skipping tool_call #%d: invalid JSON (%s)", i, e
                )
                continue
            # (optional) sanity checks
            if (
                not isinstance(obj, dict)
                or "name" not in obj
                or "arguments" not in obj
            ):
                logging.warning(
                    "Skipping tool_call #%d: missing 'name' or 'arguments': %r",
                    i,
                    obj,
                )
                continue
            tools.append(obj)
        return tools

    def _prompt_mode_post_parsing(self, raw_response: str) -> Dict[str, Any]:
        # Extract reasoning content if any.
        parsed_response = self._maybe_extract_reasoning_content(raw_response)
        reasoning_content = parsed_response["reasoning_content"]
        response_content = parsed_response["response_content"]

        # Extract tool calls if any.
        tool_calls = self._extract_tool_calls(response_content)
        if tool_calls:
            dialogue = {
                "tool_calls": tool_calls,
                "reasoning_content": reasoning_content,
            }
        else:
            dialogue = {
                "content": response_content,
                "reasoning_content": reasoning_content,
            }

        return dialogue

    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: Optional[List[Any]] = NOT_GIVEN,
    ) -> Dict[str, Any]:
        # Obtain the raw response from the LLM.
        message = self._generate(
            messages=messages,
            generation_kwargs=generation_kwargs,
            functions=functions,
        )

        # Return the dialogue to the Scene.
        # It contains at least 3 fields: "sender", "recipient", "message".
        dialogue: Dict[str, Any] = {}

        if self.fc_mode:
            message_text = message.content
            parsed_text = self._maybe_extract_reasoning_content(message_text)
            reasoning_content = parsed_text["reasoning_content"]
            response_content = parsed_text["response_content"]

            dialogue.update(
                {
                    "sender": "agent",
                    "reasoning_content": reasoning_content,
                    "response_content": response_content,
                }
            )

            if message.tool_calls:
                # If the agent calls any tools, pass the dialogue to an executor.
                tool_calls = convert_fc_namespace_to_dict(message.tool_calls)
                dialogue.update(
                    {
                        "recipient": "executor",
                        "message": {
                            "role": "assistant",
                            "tool_calls": tool_calls,
                        },
                    }
                )
            else:
                # If the agent doesn't call any tools, return the dialogue to the user.
                dialogue.update(
                    {
                        "recipient": "user",
                        "message": {
                            "role": "assistant",
                            "content": response_content,
                        },
                    }
                )

        else:
            parsed_message = self._prompt_mode_post_parsing(
                raw_response=message.content
            )
            if "tool_calls" in parsed_message:
                dialogue.update(
                    {
                        "sender": "agent",
                        "recipient": "executor",
                        "reasoning_content": parsed_message[
                            "reasoning_content"
                        ],
                        "response_content": parsed_message["response_content"],
                        "message": {
                            "role": "assistant",
                            "tool_calls": parsed_message["tool_calls"],
                        },
                    }
                )
            else:
                dialogue.update(
                    {
                        "sender": "agent",
                        "recipient": "user",
                        "reasoning_content": parsed_message[
                            "reasoning_content"
                        ],
                        "response_content": parsed_message["response_content"],
                        "message": {
                            "role": "assistant",
                            "content": parsed_message["content"],
                        },
                    }
                )

        return dialogue
