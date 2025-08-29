import inspect
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Tuple

from .scenarios.scenariosen.phone_platform.base_api import BaseApi
from .scenarios.scenariosen.phone_platform.food_services import FoodPlatform
from .scenarios.scenariosen.phone_platform.message import MessageApi
from .scenarios.scenariosen.phone_platform.reminder import ReminderApi
from .scenarios.scenariosen.travel import Travel
from .scenarios.scenarioszh.phone_platform.base_api import (
    BaseApi as BaseApi_Zh,
)
from .scenarios.scenarioszh.phone_platform.food_services import (
    FoodPlatform as FoodPlatform_Zh,
)
from .scenarios.scenarioszh.phone_platform.message import (
    MessageApi as MessageApi_Zh,
)
from .scenarios.scenarioszh.phone_platform.reminder import (
    ReminderApi as ReminderApi_Zh,
)
from .scenarios.scenarioszh.travel import Travel as Travel_Zh
from .scenarios.utils import SAVED_CLASS

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
_SENTINEL = object()
DEBUG = os.environ.get("DEBUG", False)


def iter_instance_attrs(obj: object) -> Iterator[Tuple[str, Any]]:
    """
    Yield (name, value) for all per-instance attributes (i.e., set on self),
    supporting both __dict__ and __slots__. Skips callables and private names.
    """
    names = set()

    # Attributes stored in __dict__
    if hasattr(obj, "__dict__"):
        names.update(obj.__dict__.keys())

    # Attributes stored in __slots__ (walk the MRO to include bases)
    for cls in obj.__class__.mro():
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.update(slots or ())

    for name in sorted(n for n in names if n and not n.startswith("_")):
        # Prefer __dict__ value when available (avoids triggering properties)
        val = _SENTINEL
        if hasattr(obj, "__dict__"):
            val = obj.__dict__.get(name, _SENTINEL)
        if val is _SENTINEL:
            # Might be a slot; grab it (and ignore missing/descriptor errors)
            try:
                val = getattr(obj, name)
            except Exception:
                continue

        if callable(val):
            continue
        yield name, val


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


def parse_func_args(func_args: str) -> Dict[str, Any]:
    try:
        arguments_dict = json.loads(func_args)
    except json.JSONDecodeError as e:
        logging.warning(
            f"Broken tool_call: invalid JSON {e}\nArgs: \n{func_args}\n"
        )
        arguments_dict = {}

    return arguments_dict


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
            CLASS_MAPPING_EN.copy()
            if language == "en"
            else CLASS_MAPPING_ZH.copy()
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

    def func_to_callable_classes(self, func_name):
        # Create a function names to class name map.
        # This is because the function descriptions don't have their class, only method (function) names
        callable_class_names = []
        for cls_name in self.exe_classes.keys():
            cls = (
                CLASS_MAPPING_EN[cls_name]
                if self.language == "en"
                else CLASS_MAPPING_ZH[cls_name]
            )
            if hasattr(cls, func_name):
                callable_class_names.append(cls_name)

        return callable_class_names

    def call_functions(
        self, functions: List[Dict[str, str | dict]]
    ) -> List[str]:
        """
        functions: List of functions to call. Each element is a dictionary with "name" and "arguments" keys
        return: List of function outputs in text format.
        """
        func_output_list = []
        for function in functions:
            func_name: str = function["name"]
            func_args: dict = parse_func_args(function["arguments"])

            # One function may be shared across different classes.
            # Call all the class methods, if the name matches.
            callable_cls_names = self.func_to_callable_classes(func_name)
            for cls_name in callable_cls_names:
                cls_instance = self.exe_classes[cls_name]
                cls_method = getattr(cls_instance, func_name, None)
                if cls_method is not None:
                    try:
                        func_output = cls_method(**func_args)
                    except Exception as e:
                        logging.warning(f"Invalid function call: {e}\n")
                        func_output = (
                            f"Function call failed because of exception: {e}"
                        )
                    func_output = str(func_output)
                else:
                    func_output = (
                        f"Function {func_name} not found in Class {cls_name}."
                    )

                func_output_list.append(func_output)

        return func_output_list

    def get_exe_class(self, involved_class: str):
        return self.exe_classes[involved_class]

    def return_exe_class_status(self) -> List[Dict[str, Dict[str, Any]]]:
        status_list = []
        for involved_class in self.involved_classes:
            exe_class = self.get_exe_class(involved_class)
            status_dict = {}
            for attr_name in SAVED_CLASS[involved_class]:
                status = getattr(exe_class, attr_name)
                status_dict.update({attr_name: status})

            status_list.append({involved_class: status_dict})

        return status_list

    def check_classes(self):
        for idx, obj in enumerate(self.exe_classes.values(), 1):
            print(f"Instance {idx} ({obj.__class__.__name__}):")
            for k, v in iter_instance_attrs(obj):
                print(f"  {k} = {v!r}")
