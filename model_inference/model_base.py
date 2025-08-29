from typing import Any, Dict, List

from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletionMessage


class BaseModelInference:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
    ):
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: List[Any] | None | type(NOT_GIVEN) = NOT_GIVEN,
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
        functions: A list of functions/tools to use.
            [
                {
                    "type": "function",
                    "function": {
                        "name": "my_func",
                        "description": "",
                        "parameters": {},
                    }
                },
            ]
        """
        temperature = generation_kwargs.get("temperature", 0.1)
        max_tokens = generation_kwargs.get("max_tokens", 64)
        top_p = generation_kwargs.get("top_p", 0.9)

        # tool_choice="auto": let the model choose between function calls or text outputs.

        chat_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice=NOT_GIVEN if functions == NOT_GIVEN else "auto",
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
        functions: List[Any] | None | type[NOT_GIVEN] = NOT_GIVEN,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Implement this `generate`.")

    def tool_output_to_message(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_outputs: List[str],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement this `tool_output_to_message`.")
