from model_inference.multi_turn.multi_turn_utils import *


class Executor:
    def __init__(
        self,
        class_init_config,
        language,
    ) -> None:
        self.class_init_config = class_init_config
        self.language = language

    def respond(self, history) -> None:
        current_message = {}
        message = history[-1]["message"]

        function_call_list = self.decode_function_list(message)

        single_turn_execution_results, result_instances = (
            execute_agent_func_call(
                func_call_list=function_call_list,
                initial_config=self.initial_config,
                involved_classes=self.involved_classes,
                model_name=self.agent_model_name,
                test_entry_id=self.test_id,
                language=self.language,
            )
        )

        parsed_results = []
        for item in single_turn_execution_results:
            try:
                parsed_item = json.loads(item)
                parsed_results.append(parsed_item)
            except json.JSONDecodeError:
                parsed_results.append(item)

        current_message["sender"] = "execution"
        current_message["recipient"] = "agent"
        current_message["message"] = parsed_results

        return current_message, result_instances
