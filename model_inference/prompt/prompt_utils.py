import importlib
from typing import Dict, List

from .prompt_map import PROMPT_NAME_MAP


def compose_agent_system_prompt(
    category: str,
    involved_classes: List[str],
    language: str,
    model_name: str,
) -> str:
    prompt_lib = importlib.import_module(PROMPT_NAME_MAP[model_name][language])

    if language == "zh":
        agent_system_prompt = (
            prompt_lib.MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
            if category == "multi_turn"
            else prompt_lib.MULTI_STEP_AGENT_PROMPT_SYSTEM_ZH
        )
        if "Travel" in involved_classes:
            agent_system_prompt += prompt_lib.TRAVEL_PROMPT_ZH
        if "BaseApi" in involved_classes:
            agent_system_prompt += prompt_lib.BASE_PROMPT_ZH
    elif language == "en":
        agent_system_prompt = (
            prompt_lib.MULTI_TURN_AGENT_SYSTEM_PROMPT_EN
            if category == "multi_turn"
            else prompt_lib.MULTI_STEP_AGENT_SYSTEM_PROMPT_EN
        )
        if "Travel" in involved_classes:
            agent_system_prompt += prompt_lib.TRAVEL_PROMPT_EN
        if "BaseApi" in involved_classes:
            agent_system_prompt += prompt_lib.BASE_PROMPT_EN
    else:
        raise KeyError("Language not supported.")

    return agent_system_prompt


def compose_user_system_message(
    instruction: str,
    involved_classes: List[str],
    language: str,
    model_name: str,
) -> List[Dict[str, str]]:
    prompt_lib = importlib.import_module(PROMPT_NAME_MAP[model_name][language])

    if language == "zh":
        if "Travel" in involved_classes:
            system_prompt = prompt_lib.USER_SYSTEM_PROMPT_TRAVEL_ZH.format(instruction=instruction)
        else:
            system_prompt = prompt_lib.USER_SYSTEM_PROMPT_BASE_ZH.format(instruction=instruction)
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": "今天有什么需要帮助的吗？"},
        ]
    elif language == "en":
        if "Travel" in involved_classes:
            system_prompt = prompt_lib.USER_SYSTEM_PROMPT_TRAVEL_EN.format(instruction=instruction)
        else:
            system_prompt = prompt_lib.USER_SYSTEM_PROMPT_BASE_EN.format(instruction=instruction)
        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": "How can I help you today?"},
        ]
    else:
        raise KeyError("Language not supported.")


def compose_preference_system_prompt(
    profile,
    language: str,
    model_name: str,
):
    prompt_lib = importlib.import_module(PROMPT_NAME_MAP[model_name][language])

    if language == "zh":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(profile=profile)
    elif language == "en":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(profile=profile)
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_special_system_prompt(
    time,
    language: str,
    model_name: str,
):
    prompt_lib = importlib.import_module(PROMPT_NAME_MAP[model_name][language])

    if language == "zh":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(time=time)
    elif language == "en":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(time=time)
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_normal_system_prompt(
    time,
    language: str,
    model_name: str,
):
    prompt_lib = importlib.import_module(PROMPT_NAME_MAP[model_name][language])

    if language == "zh":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time=time)
    elif language == "en":
        system_prompt = prompt_lib.SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(time=time)
    else:
        raise KeyError("Language not supported.")
    return system_prompt
