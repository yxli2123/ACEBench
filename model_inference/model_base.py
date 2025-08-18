import subprocess
import sys
from typing import Any, Dict, List, Optional

from openai import NOT_GIVEN, OpenAI
from openai.types.chat import ChatCompletionMessage


class BaseModelInference:
    def __init__(
        self,
        model_name: str,
        vllm_kwargs: Optional[Dict[str, str | int]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        fc_mode: bool = False,
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
        functions: List[Any] | None | NOT_GIVEN = NOT_GIVEN,
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
        functions: List[Any] | None | NOT_GIVEN = NOT_GIVEN,
    ) -> Dict[str, Any]:
        raise NotImplementedError("Implement this `generate`.")

    def tool_output_to_message(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_outputs: List[str],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Implement this `tool_output_to_message`.")
