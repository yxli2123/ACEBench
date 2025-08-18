from typing import Any, Callable, Dict, List

from tqdm import tqdm

from model_inference.executor import Executor
from model_inference.model_agent import tool_output_to_message
from model_inference.model_base import BaseModelInference
from model_inference.multi_step.common_agent_step import CommonAgent_Step
from model_inference.multi_step.execution_role_step import EXECUTION_STEP
from model_inference.multi_step.multi_step_scene import Mulit_Step_Scene
from model_inference.multi_turn.APIModel_user import (
    SYSTEM_PROMPT_BASE_ZH,
    SYSTEM_PROMPT_TRAVEL_ZH,
)
from model_inference.prompt.prompt_en import *
from model_inference.prompt.prompt_zh import *

SAVED_CLASS = {
    "BaseApi": ["wifi", "logged_in"],
    "MessageApi": ["inbox"],
    "ReminderApi": ["reminder_list"],
    "FoodPlatform": ["users", "logged_in_users", "orders"],
    "Finance": [
        "user_accounts",
        "is_logged_in",
        "deposit_history",
        "withdrawal_history",
        "loan_history",
        "orders",
        "holdings",
    ],
    "Travel": ["users", "reservations"],
}


class InferenceScene:
    def __init__(
        self,
        question: str,
        functions: List[Callable],
        max_dialog_turns=40,
        language="zh",
        generation_kwargs: Dict[str, Any] = None,
        user_generation_kwargs: Dict[str, Any] = None,
    ):
        self.question = question
        self.functions = functions
        self.max_dialog_turns = max_dialog_turns
        self.language = language
        self.generation_kwargs = generation_kwargs
        self.user_generation_kwargs = (
            user_generation_kwargs
            if user_generation_kwargs is not None
            else generation_kwargs
        )

    def init_agent_system_prompt(
        self,
        category: str,
        involved_classes: List[str],
    ):
        if self.language == "zh":
            agent_system_prompt = (
                MULTI_TURN_AGENT_PROMPT_SYSTEM_ZH
                if category == "multi_turn"
                else MULTI_STEP_AGENT_PROMPT_SYSTEM_ZH
            )
            if "Travel" in involved_classes:
                agent_system_prompt += TRAVEL_PROMPT_ZH
            if "BaseApi" in involved_classes:
                agent_system_prompt += BASE_PROMPT_ZH
        elif self.language == "en":
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

    def inference(
        self, question, functions, time, profile, test_case, test_id: str
    ):
        category = test_id.rsplit("_", 1)[0]
        if "multi_turn" in category and "agent" in category:
            initial_config = test_case["initial_config"]
            involved_classes = test_case["involved_classes"]
            test_id = test_case["id"].split("_")[-1]
            result, process_list = self.multi_turn_inference(
                question,
                initial_config,
                functions,
                involved_classes,
                test_id,
                time,
            )
            return result, process_list
        elif "multi_step" in category:
            initial_config = test_case["initial_config"]
            involved_classes = test_case["involved_classes"]
            test_id = test_case["id"].split("_")[-1]
            result, process_list = self.multi_step_inference(
                question,
                initial_config,
                functions,
                involved_classes,
                test_id,
                time,
            )
            return result, process_list
        else:
            result = self.single_turn_inference(
                question, functions, time, profile, id
            )

        return result

    def single_turn_inference(self, question, functions, time, profile, id):
        category = id.rsplit("_", 1)[0]
        if self.language == "zh":
            if "special" in category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH.format(
                    time=time, function=functions
                )
            elif "preference" in category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_ZH.format(
                    profile=profile, function=functions
                )
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(
                    time=time, function=functions
                )
            user_prompt = USER_PROMPT_ZH.format(question=question)

        elif self.language == "en":
            if "special" in category:
                system_prompt = SYSTEM_PROMPT_FOR_SPECIAL_DATA_EN.format(
                    time=time, function=functions
                )

            elif "preference" in category:
                system_prompt = SYSTEM_PROMPT_FOR_PREFERENCE_DATA_EN.format(
                    profile=profile, function=functions
                )
            else:
                system_prompt = SYSTEM_PROMPT_FOR_NORMAL_DATA_EN.format(
                    time=time, function=functions
                )
            user_prompt = USER_PROMPT_EN.format(question=question)

        result = self.model.inference(system_prompt, user_prompt)
        return result

    def multi_turn_inference(
        self,
        user_model: BaseModelInference,
        agent_model: BaseModelInference,
        question: str,
        functions: List[Dict[str, Any]],
        initial_config: Dict[str, Any],
        involved_classes: List[str],
    ):
        # Turns |            Signals                  |
        # ------|-------------------------------------|
        #   1   |    user          -->  agent         |
        #   2   |    agent         -->  executor/user |
        #   3   |    executor/user -->  agent         |
        #   4   |              ....                   |
        #   N   |    agent         -->  user          |

        user_message_history = []
        agent_message_history = []
        dialogue_history = []

        # Initialize user messages.
        if self.language == "zh":
            if "Travel" in involved_classes:
                user_system_prompt = SYSTEM_PROMPT_TRAVEL_ZH
            else:
                user_system_prompt = SYSTEM_PROMPT_BASE_ZH
            user_user_prompt = "今天有什么需要帮助的吗？"

        elif self.language == "en":
            if "Travel" in involved_classes:
                user_system_prompt = SYSTEM_PROMPT_TRAVEL_EN
            else:
                user_system_prompt = SYSTEM_PROMPT_BASE_EN
            user_user_prompt = "How can I assist you today?"
        else:
            raise KeyError("Language not supported.")

        user_message_history.extend(
            [
                {
                    "role": "system",
                    "content": user_system_prompt.format(instruction=question),
                },
                {
                    "role": "user",
                    "content": user_user_prompt,
                },
            ]
        )

        # Generate initial query from the user model.
        user_init_query = user_model.generate(
            user_message_history, self.user_generation_kwargs
        )["message"]
        dialogue_history.append(
            {
                "sender": "user",
                "recipient": "agent",
                "message": user_init_query,
            }
        )

        # Initialize agent messages.
        agent_system_prompt = self.init_agent_system_prompt(
            category="multi_turn",
            involved_classes=involved_classes,
        )
        agent_message_history.append(
            {
                "role": "system",
                "content": agent_system_prompt,
            }
        )

        # Initialize the executor.
        executor = Executor(
            involved_classes=involved_classes,
            class_init_config=initial_config,
            language=self.language,
        )

        for turn_id in range(self.max_dialog_turns):
            current_recipient = dialogue_history[-1]["recipient"]
            current_sender = dialogue_history[-1]["sender"]
            if current_recipient == "user":
                user_message_history.append(
                    {
                        "role": "user",
                        "content": dialogue_history[-1]["message"],
                    }
                )

                # Stop the dialogue if 'finish conversation' is detected.
                if "finish conversation" in dialogue_history[-1]["message"]:
                    break

                current_dialogue = user_model.generate(
                    messages=agent_message_history,
                    generation_kwargs=self.generation_kwargs,
                )
                dialogue_history.append(current_dialogue)
            elif current_recipient == "agent":
                if current_sender == "user":
                    agent_message_history.append(
                        {
                            "role": "user",
                            "content": dialogue_history[-1]["message"],
                        }
                    )
                elif current_sender == "executor":
                    agent_message_history.extend(
                        dialogue_history[-1]["message"]
                    )
                current_dialogue = agent_model.generate(
                    messages=agent_message_history,
                    generation_kwargs=self.generation_kwargs,
                    functions=functions,
                )
                dialogue_history.append(current_dialogue)
            elif current_recipient == "executor":
                tool_calls = dialogue_history[-1]["tool_calls"]
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
            else:
                raise ValueError(
                    "Unknown recipient type. Only `user`, `agent` , and `executor` are supported."
                )

        # Obtain the attributes of the involved classes as the final results for evaluation.
        result_list = []
        for involved_class in involved_classes:
            exe_class = executor.get_exe_class(involved_class)
            attr_name_list = SAVED_CLASS[involved_class]
            result_list.append(
                {
                    involved_class: {
                        attr_name: getattr(exe_class, attr_name)
                        for attr_name in attr_name_list
                    }
                }
            )

        return result_list, dialogue_history

    def multi_step_inference(
        self,
        agent_model: BaseModelInference,
        question: str,
        functions: List[Dict[str, Any]],
        initial_config: Dict[str, Any],
        involved_classes: List[str],
    ):
        # Generate initial query from the user model.
        dialogue_history = [
            {
                "sender": "user",
                "recipient": "agent",
                "message": question,
            }
        ]

        # Initialize agent messages.
        agent_message_history = []
        agent_system_prompt = self.init_agent_system_prompt(
            category="multi_step",
            involved_classes=involved_classes,
        )
        agent_message_history.append(
            {
                "role": "system",
                "content": agent_system_prompt,
            }
        )

        # Initialize the executor.
        executor = Executor(
            involved_classes=involved_classes,
            class_init_config=initial_config,
            language=self.language,
        )

        for turn_id in range(self.max_dialog_turns):
            current_recipient = dialogue_history[-1]["recipient"]
            current_sender = dialogue_history[-1]["sender"]
            if current_recipient == "user":
                # Stop the dialogue if 'finish conversation' is detected.
                if "finish conversation" in dialogue_history[-1]["message"]:
                    break

            elif current_recipient == "agent":
                if current_sender == "user":
                    agent_message_history.append(
                        {
                            "role": "user",
                            "content": dialogue_history[-1]["message"],
                        }
                    )
                elif current_sender == "executor":
                    agent_message_history.extend(
                        dialogue_history[-1]["message"]
                    )
                current_dialogue = agent_model.generate(
                    messages=agent_message_history,
                    generation_kwargs=self.generation_kwargs,
                    functions=functions,
                )
                dialogue_history.append(current_dialogue)
            elif current_recipient == "executor":
                tool_calls = dialogue_history[-1]["tool_calls"]
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
            else:
                raise ValueError(
                    "Unknown recipient type. Only `user`, `agent` , and `executor` are supported."
                )

        # Obtain the attributes of the involved classes as the final results for evaluation.
        result_list = []
        for involved_class in involved_classes:
            exe_class = executor.get_exe_class(involved_class)
            attr_name_list = SAVED_CLASS[involved_class]
            result_list.append(
                {
                    involved_class: {
                        attr_name: getattr(exe_class, attr_name)
                        for attr_name in attr_name_list
                    }
                }
            )

        return result_list, dialogue_history

    def multi_step_inference_legacy(
        self,
        question,
        initial_config,
        functions,
        involved_classes,
        test_id,
        time,
    ):
        agent = CommonAgent_Step(
            model=self.model, language=self.language, functions=functions
        )
        execution = EXECUTION_STEP(
            agent_model_name=self.model_name,
            initial_config=initial_config,
            involved_classes=involved_classes,
            test_id=test_id,
            language=self.language,
        )

        scene = Mulit_Step_Scene(
            question=question,
            initial_state=initial_config,
            functions=functions,
            agent_role=agent,
            language=self.language,
        )

        message_history = scene.dialogue_history
        result_list = []

        result_instance_list = []
        mile_stone = []
        with tqdm(
            total=self.max_message_index, desc="Processing Messages"
        ) as pbar:
            for index in range(self.max_message_index):
                last_sender = message_history[-1]["sender"]
                if index == 0 or last_sender == "execution":
                    inference_message = scene.get_inference_message()
                    current_message = agent.respond(inference_message)
                else:
                    current_message, result_instance = execution.respond(
                        message_history
                    )
                    mile_stone_message = message_history[-1]["message"]
                    mile_stone.append(mile_stone_message)
                    if result_instance not in result_instance_list:
                        result_instance_list.append(result_instance)

                scene.add_dialogue(current_message)

                if (
                    index > 1
                    and "finish conversation" in current_message["message"]
                ):
                    break
                pbar.update(1)

        scene.write_message_history(test_id, self.model_name)

        for result_instance in result_instance_list:
            for name, instance in result_instance.items():
                item_dict = {}
                for item in instance.__dict__:
                    if item in SAVED_CLASS[name]:
                        item_dict[item] = instance.__dict__[item]
                result_list.append({name: item_dict})

        # Return the instance name - its properties will be tested against expectations later
        return result_list, mile_stone
