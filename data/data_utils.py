import json
import os
import re
import warnings
from typing import Dict, List

from model_inference.utils import pystr_to_calls, wrap_tool_protocol

DEBUG = os.environ.get("DEBUG", False)


def convert_text_to_messages(text: str) -> List[Dict[str, str]]:
    # Regex: match 'user:' or 'system:' and capture their role + content
    pattern = re.compile(
        r"(user|system):\s*(.*?)(?=(?:\n(?:user|system):)|$)", re.DOTALL
    )

    messages = []
    for _role, _content in pattern.findall(text):
        content = _content.strip()
        role = "user" if "user" in _role else "assistant"

        # TODO: Change this lazy function match to real function match
        if content.startswith("[") and content.endswith("]"):
            cont_key = "tool_calls"
            # Convert string like list of json to real List[Dict[str, Any]] type
            try:
                content = pystr_to_calls(content)
                content = wrap_tool_protocol(content)
                for c in content:
                    c.update({"id": str(id(c))})
                    c["function"]["arguments"] = json.dumps(
                        c["function"]["arguments"]
                    )
            except Exception:
                cont_key = "content"
                warnings.warn(f"Could not convert to dict.\n{content}")

        else:
            cont_key = "content"

        messages.append(
            {
                "role": role,
                cont_key: content,
            }
        )

    return messages


def maybe_rm_role_in_text(text: str, role: str = "user") -> str:
    if text.startswith(f"{role}:"):
        return text[len(f"{role}:") :].strip()

    return text
