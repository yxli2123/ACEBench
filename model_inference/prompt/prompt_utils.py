from typing import List

from prompt_en import *
from prompt_zh import *


def compose_agent_system_prompt(
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


def compose_user_system_prompt():
    pass


def compose_preference_system_prompt(
    profile,
    functions,
    language,
):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(
            profile=profile, function=functions
        )
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(
            profile=profile, function=functions
        )
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_special_system_prompt(
    time,
    functions,
    language,
):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(
            time=time, function=functions
        )
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(
            time=time, function=functions
        )
    else:
        raise KeyError("Language not supported.")
    return system_prompt


def compose_normal_system_prompt(
    time,
    functions,
    language,
):
    if language == "zh":
        system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(
            time=time, function=functions
        )
    elif language == "en":
        system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(
            time=time, function=functions
        )
    else:
        raise KeyError("Language not supported.")
    return system_prompt
