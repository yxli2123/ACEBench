import inspect
from copy import deepcopy
from typing import Any, Dict, List

from scenarios.scenariosen.phone_platform.base_api import BaseApi
from scenarios.scenariosen.phone_platform.food_services import FoodPlatform
from scenarios.scenariosen.phone_platform.message import MessageApi
from scenarios.scenariosen.phone_platform.reminder import ReminderApi
from scenarios.scenariosen.travel import Travel
from scenarios.scenarioszh.phone_platform.base_api import BaseApi as BaseApi_Zh
from scenarios.scenarioszh.phone_platform.food_services import (
    FoodPlatform as FoodPlatform_Zh,
)
from scenarios.scenarioszh.phone_platform.message import (
    MessageApi as MessageApi_Zh,
)
from scenarios.scenarioszh.phone_platform.reminder import (
    ReminderApi as ReminderApi_Zh,
)
from scenarios.scenarioszh.travel import Travel as Travel_Zh

CLASS_MAPPING_EN = {
    "Travel": Travel,
    "BaseApi": BaseApi,
    "FoodPlatform": FoodPlatform,
    "MessageApi": MessageApi,
    "ReminderApi": ReminderApi,
}

CLASS_MAPPING_ZH = {
    "Travel": Travel_Zh,
    "BaseApi": BaseApi_Zh,
    "FoodPlatform": FoodPlatform_Zh,
    "MessageApi": MessageApi_Zh,
    "ReminderApi": ReminderApi_Zh,
}


def create_map_function_to_class(cls) -> Dict[str, str]:
    """
    cls: Input class.
    It creates a map of function names to their class name.
    For example:
    {
       "turn_on_wifi": "BaseApi",
       "login_device": "BaseApi",
    }
    """

    members = inspect.getmembers(cls, predicate=inspect.isroutine)
    # Get all the non-private methods.
    fun_names = [name for name, _ in members if not name.startswith("_")]
    # keep only those defined directly on this class (not inherited)
    fun_names = set([n for n in fun_names if n in cls.__dict__])

    return {k: cls.__name__ for k in fun_names}


class Executor:
    """This class execute the function calls from the agent and return the function output back to the agent."""

    def __init__(
        self,
        class_init_config: Dict[str, Any],
        involved_classes: List[str],
        language: str,
    ) -> None:
        """
        Initialization creates the involved classes, for example, BaseApi, MessageApi, ReminderApi.
        """
        _class_init_config = deepcopy(class_init_config)
        self.involved_classes = deepcopy(involved_classes)
        self.language = language

        class_mapping = (
            CLASS_MAPPING_EN if language == "en" else CLASS_MAPPING_EN.copy()
        )

        # Instantiate the involved classes.
        # Use a dict to maintain the self.exe_classes, cls_name: cls_instance
        self.exe_classes = {
            cls_name: class_mapping[cls_name]()
            for cls_name in involved_classes
        }

        # Initialize the instantiated classes with default BaseApi config first.
        base_api_config = _class_init_config.pop("BaseApi", {})
        for cls_instance in self.exe_classes.values():
            cls_instance._load_scenario(base_api_config)

        # Initialize the instantiated classes with their own config if any.
        for cls_name, cls_config in _class_init_config.items():
            self.exe_classes[cls_name]._load_scenario(cls_config)

        # Create a function names to class name map.
        # This is because the function descriptions don't have their class, only method (function) names
        self.func_to_class = {}
        for cls_name in self.exe_classes.keys():
            func_to_class = create_map_function_to_class(
                class_mapping[cls_name]
            )
            self.func_to_class.update(func_to_class)

    def call_functions(
        self, functions: List[Dict[str, str | dict]]
    ) -> List[str]:
        """
        functions: List of functions to call. Each element is a dictionary with "name" and "arguments" keys
        return: List of function outputs in text format.
        """
        func_output_list = []
        for function in functions:
            func_name = function["name"]
            func_args = function["arguments"]

            cls_name = self.func_to_class[func_name]
            cls_instance = self.exe_classes[cls_name]
            cls_method = getattr(cls_instance, func_name, None)
            if cls_method is not None:
                func_output = getattr(cls_instance, func_name)(*func_args)
                func_output = str(func_output)
            else:
                func_output = (
                    f"Function {func_name} not found in Class {cls_name}."
                )

            func_output_list.append(func_output)

        return func_output_list

    def get_exe_class(self, involved_class: str):
        return self.exe_classes[involved_class]
