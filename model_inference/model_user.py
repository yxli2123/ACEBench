from typing import Any, Dict, List

from openai import NOT_GIVEN

from .model_base import BaseModelInference


class UserModelInference(BaseModelInference):
    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_kwargs: Dict[str, Any],
        functions: List[Any] | None | type[NOT_GIVEN] = NOT_GIVEN,
        **kwargs,
    ) -> Dict[str, Any]:
        # Obtain the raw response from the LLM.
        message = self._generate(
            messages=messages,
            generation_kwargs=generation_kwargs,
        )

        # Return the dialogue to the Scene.
        # It contains at least 3 fields: "sender", "recipient", "message".
        dialogue = {
            "sender": "user",
            "recipient": "agent",
            "message": message.content,
        }

        return dialogue
