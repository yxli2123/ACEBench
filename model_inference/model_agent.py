import json
import logging
import re
from typing import Any, Dict, List, Optional

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageToolCall

from model_inference.model_base import BaseModelInference

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
        functions: List[Any] | None | NOT_GIVEN = NOT_GIVEN,
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
                # [{"name": function_name, "arguments": arguments_dict}]
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
