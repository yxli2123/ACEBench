from typing import Dict, List

from .prompt_en import *
from .prompt_zh import *


def compose_agent_system_prompt_fc(
    category: str,
    involved_classes: List[str],
    language: str,
) -> str:
    if language == "zh":
        agent_system_prompt = (
            MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
            if category == "multi_turn"
            else MULTI_STEP_AGENT_PROMPT_SYSTEM_ZH
        )
        if "Travel" in involved_classes:
            agent_system_prompt += TRAVEL_PROMPT_ZH
        if "BaseApi" in involved_classes:
            agent_system_prompt += BASE_PROMPT_ZH
    elif language == "en":
        agent_system_prompt = (
            MULTI_TURN_AGENT_PROMPT_SYSTEM_EN
            if category == "multi_turn"
            else MULTI_STEP_AGENT_PROMPT_SYSTEM_EN
        )
        if "Travel" in involved_classes:
            agent_system_prompt += TRAVEL_PROMPT_EN
        if "BaseApi" in involved_classes:
            agent_system_prompt += BASE_PROMPT_EN
    else:
        raise KeyError("Language not supported.")

    return agent_system_prompt


def compose_user_system_message_fc(
    role: str, language: str
) -> List[Dict[str, str]]:
    if language == "zh":
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_USER_MODEL_ZH.format(role=role),
            },
            {"role": "user", "content": "今天有什么需要帮助的吗？"},
        ]
    elif language == "en":
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT_USER_MODEL_EN.format(role=role),
            },
            {"role": "user", "content": "How can I help you today?"},
        ]
    else:
        raise KeyError("Language not supported.")


def compose_preference_system_prompt_fc(profile, language):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(
            profile=profile
        )
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(
            profile=profile
        )
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_special_system_prompt_fc(time, language):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time=time)
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time)
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_normal_system_prompt_fc(time, language):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time=time)
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time)
    else:
        raise KeyError("Language not supported.")
    return system_prompt
