import json
import logging
import os
import warnings
from typing import Any, Dict, List

from openai import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageToolCall

from .model_base import BaseModelInference
from .utils import pystr_to_calls, wrap_tool_protocol

logging.basicConfig(level=logging.WARNING)

DEBUG = os.environ.get("DEBUG", False)


class BaseAgentInference(BaseModelInference):
    @staticmethod
    def convert_fc_namespace_to_dict(
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> List[Dict[str, str]]:
        tool_calls_in_dict = []
        for i, tool_call in enumerate(tool_calls):
            function = tool_call.function
            arguments = function.arguments
            call_id = getattr(
                tool_call, "id", f"{function.name}-{id(arguments)}"
            )

            tool_calls_in_dict.append(
                {"name": function.name, "arguments": arguments, "id": call_id}
            )

        return tool_calls_in_dict

    def pre_process(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Any] | None | type(NOT_GIVEN) = NOT_GIVEN,
        **kwargs,
    ):
        return messages, functions

    def post_process(self, text: None | str, **kwargs) -> Dict[str, Any]:
        return {"response_content": text}

    def has_tool_calls(self, message, parsed_text) -> bool:
        if message.tool_calls:
            return True
        else:
            return False

    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: List[Any] | None | type(NOT_GIVEN) = NOT_GIVEN,
        **kwargs,
    ) -> Dict[str, Any]:
        messages, functions = self.pre_process(messages, functions, **kwargs)

        # Obtain the raw response from the LLM.
        message = self._generate(
            messages=messages,
            generation_kwargs=generation_kwargs,
            functions=functions,
        )

        # Return the dialogue to the Scene.
        # It contains at least 3 fields: "sender", "recipient", "message".
        dialogue: Dict[str, Any] = {"sender": "agent"}

        message_text: str | None = message.content
        parsed_text = self.post_process(message_text, **kwargs)
        dialogue.update(parsed_text)

        if self.has_tool_calls(message, parsed_text):
            # If the agent calls any tools, pass the dialogue to an executor.
            # [{"name": function_name, "arguments": arguments_in_json_str}]
            tool_calls = self.convert_fc_namespace_to_dict(message.tool_calls)

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
            response_content = parsed_text.get(
                "response_content", message_text
            )
            dialogue.update(
                {
                    "recipient": "user",
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                }
            )

        return dialogue


class Qwen3AgentInference(BaseAgentInference):
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
    def _extract_tool_calls(fc_pystr: str) -> List[Dict[str, Any]]:
        tools = []
        try:
            fc_dict_list = pystr_to_calls(fc_pystr)
        except Exception as e:
            warnings.warn(f"Failed to convert pystr to calls\n{e}")
            return []

        for fc_dict in fc_dict_list:
            fc_dict["arguments"] = json.dumps(fc_dict["arguments"])
            fc_dict.update(
                {"id": f"{fc_dict['name']}-{id(fc_dict['arguments'])}"}
            )
        tools.extend(fc_dict_list)
        return tools

    def post_process(self, text: str, **kwargs) -> Dict[str, Any]:
        # Extract reasoning content if any.
        parsed_response = self._maybe_extract_reasoning_content(text)
        reasoning_content = parsed_response["reasoning_content"]
        response_content = parsed_response["response_content"]

        # Extract tool calls if any.
        tool_calls = self._extract_tool_calls(response_content)

        if tool_calls:
            dialogue = {
                "tool_calls": tool_calls,
                **parsed_response,
            }
        else:
            dialogue = {
                "content": response_content,
                **parsed_response,
            }

        return dialogue

    @staticmethod
    def _inject_function_to_sys_prompt(
        messages: List[Dict[str, Any]],
        functions: List[Any],
    ) -> List[Dict[str, Any]]:
        msg = messages[0]
        if msg["role"] == "system":
            msg["content"] = (
                msg["content"] + f"\nAPI Description:\n{functions}"
            )

        return messages

    def pre_process(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Any] | None | type(NOT_GIVEN) = NOT_GIVEN,
        **kwargs,
    ):
        messages = self._inject_function_to_sys_prompt(messages, functions)
        return messages, functions

    def has_tool_calls(self, message, parsed_text) -> bool:
        if parsed_text.get("tool_calls"):
            return True
        else:
            return False


class Qwen3AgentFCInference(BaseAgentInference):
    def pre_process(
        self,
        messages: List[Dict[str, Any]],
        functions: List[Dict[str, Any]] | None | type(NOT_GIVEN) = NOT_GIVEN,
        **kwargs,
    ):
        try:
            functions = wrap_tool_protocol(functions)
        except Exception:
            functions = []

        return messages, functions

    def post_process(self, text: str, **kwargs) -> Dict[str, str]:
        reasoning_text = ""
        if text is not None and "</think>" in text:
            parts = text.split("</think>")
            reasoning_text = (
                parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            )
            text = parts[-1].lstrip("\n")

        return {"reasoning_content": reasoning_text, "response_content": text}

    def has_tool_calls(self, message, parsed_text) -> bool:
        return True if message.tool_calls else False
