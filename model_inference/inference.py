from typing import List


class InferenceScene:
    def __init__(
        self,
        question: str,
        functions: List,
        max_dialog_turns=40,
        language="zh",
    ):
        self.question = question
        self.functions = functions
        self.max_dialog_turns = max_dialog_turns
        self.language = language
