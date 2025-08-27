from typing import Any, Dict, List

from model_inference.executor import Executor
from model_inference.model_agent import tool_output_to_message
from model_inference.model_base import BaseModelInference


def convert_messages_to_dialogue(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    dialogue = []
    for msg in messages:
        if msg["role"] == "user":
            dialogue.append(
                {
                    "sender": "user",
                    "recipient": "agent",
                    "message": msg["content"],
                }
            )
        elif msg["role"] == "assistant":
            dialogue.append(
                {
                    "sender": "agent",
                    "recipient": "user",
                    "message": msg["content"],
                }
            )

    return dialogue


def inference(
    agent_model: BaseModelInference,
    agent_system_prompt: str,
    question: str,
    functions: List[Dict[str, Any]],
    max_dialog_turns: int,
    generation_kwargs: Dict[str, Any],
    user_model_generation_kwargs: Dict[str, Any] | None = None,
    executor: Executor | None = None,
    user_model: BaseModelInference | None = None,
    user_system_prompt: str | None = None,
    agent_message_history: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """This is a general inference for agent. It is designed as agent centralized.

    # Turns |            Signals                  |
    # ------|-------------------------------------|
    #   1   |    user          -->  agent         |
    #   2   |    agent         -->  executor/user |
    #   3   |    executor/user -->  agent         |
    #   4   |              ....                   |
    #   N   |    agent         -->  user          |

    Args:
        agent_model: the agent model with system prompt injected.
        agent_system_prompt: the system prompt injected for the agent model.
        question: the question for the agent model if no user_model is provided,
            otherwise the init query for the user model.
        functions: functions that the agent model can use.
        max_dialog_turns: the maximum number of dialog turns that the agent, user, executor can make.
            1 for single turn, 16 for multi-turn and multi-step.
        generation_kwargs: the generation kwargs for the agent model.
        user_model_generation_kwargs: the user model generation kwargs.
        executor: Optional if in multi-turn or multi-step evaluation.
        user_model: Optional if in multi-turn evaluation.
        user_system_prompt: system prompt for the user model.,
        agent_message_history: For data_normal_multi_turn_user_adjust.json,
            where we know the historical multi-turn conversations.

    Returns:
        type: List[Dict[str, Any]]
        It returns the dialogue history, where the dict should contain "sender", "recipient", "message"
    """

    # Initialization
    dialogue_history: List[Dict[str, Any]] = (
        []
        if agent_message_history is None
        else convert_messages_to_dialogue(agent_message_history)
    )

    if agent_message_history is None:
        agent_message_history = [
            {"role": "system", "content": agent_system_prompt}
        ]
    elif agent_message_history[0]["role"] != "system":
        agent_message_history.insert(
            0, {"role": "system", "content": agent_system_prompt}
        )

    user_message_history = (
        [{"role": "system", "content": user_system_prompt}]
        if user_system_prompt
        else []
    )

    # Initial turn.
    # Single turn or multi step modes.
    if user_model is None:
        user_init_query = question
    # Multi-turn mode.
    else:
        user_model_generation_kwargs = (
            generation_kwargs
            if user_model_generation_kwargs is None
            else user_model_generation_kwargs
        )
        user_init_query = user_model.generate(
            user_message_history, user_model_generation_kwargs
        )["message"]
        user_message_history.append(
            {
                "role": "assistant",
                "content": user_init_query,
            }
        )

    dialogue_history.append(
        {
            "sender": "user",
            "recipient": "agent",
            "message": user_init_query,
        }
    )

    for turn_id in range(max_dialog_turns):
        current_recipient = dialogue_history[-1]["recipient"]
        current_sender = dialogue_history[-1]["sender"]
        current_message = dialogue_history[-1]["message"]

        if current_recipient == "agent":
            if current_sender == "user":
                agent_message_history.append(
                    {
                        "role": "user",
                        "content": dialogue_history[-1]["message"],
                    }
                )
            elif current_sender == "executor":
                agent_message_history.extend(dialogue_history[-1]["message"])
            current_dialogue = agent_model.generate(
                messages=agent_message_history,
                generation_kwargs=generation_kwargs,
                functions=functions,
            )
            dialogue_history.append(current_dialogue)

        elif current_recipient == "user":
            user_message_history.append(
                {
                    "role": "user",
                    "content": current_message,
                }
            )

            # Stop the dialogue if 'finish conversation' is detected.
            if "finish conversation" in current_message:
                break
            # Otherwise, continue the turn if the user model is not None (multi-turn mode).
            elif user_model is not None:
                current_dialogue = user_model.generate(
                    messages=agent_message_history,
                    generation_kwargs=user_model_generation_kwargs,
                )
                dialogue_history.append(current_dialogue)
            # Or, the model fails to output "finish conversation" for non multi-turn mode.
            # In this case, we stop the conversation manually by overriding the last dialogue.
            else:
                dialogue_history[-1] = {
                    "sender": current_sender,
                    "recipient": current_recipient,
                    "message": "finish conversation unsuccessful",
                    "original_message": current_message,
                }

        elif current_recipient == "executor":
            tool_calls = dialogue_history[-1]["tool_calls"]
            if executor:
                tool_output_list = executor.call_functions(tool_calls)
                dialogue_history.append(
                    {
                        "sender": "executor",
                        "recipient": "agent",
                        "message": tool_output_to_message(
                            tool_calls, tool_output_list
                        ),
                    }
                )
            # For single turn evaluation, there is no executor.
            else:
                break
        else:
            raise ValueError(
                "Unknown recipient type. Only `user`, `agent` , and `executor` are supported."
            )

    return dialogue_history
