import re
from typing import Dict, List


def convert_text_to_messages(text: str) -> List[Dict[str, str]]:
    # Regex: match 'user:' or 'system:' and capture their role + content
    pattern = re.compile(
        r"(user|system):\s*(.*?)(?=(?:\n(?:user|system):)|$)", re.DOTALL
    )

    messages = []
    for role, content in pattern.findall(text):
        messages.append(
            {
                "role": "user" if "user" in role else "assistant",
                "content": content.strip(),
            }
        )

    return messages
