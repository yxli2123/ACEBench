import argparse
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

from category import ACE_DATA_CATEGORY
from data.data_utils import convert_text_to_messages, maybe_rm_role_in_text
from model_inference.agent_map import AGENT_NAME_MAP
from model_inference.common_inference import inference
from model_inference.executor import Executor
from model_inference.model_base import BaseModelInference
from model_inference.model_user import UserModelInference
from model_inference.prompt.prompt_utils import (
    compose_agent_system_prompt,
    compose_normal_system_prompt,
    compose_preference_system_prompt,
    compose_special_system_prompt,
    compose_user_system_message,
)
from model_inference.utils import calls_to_pystr


def parser():
    parser = argparse.ArgumentParser(description="Generate ACEBench Results.")
    # Model name
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-8B-FC",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8181",
        help="",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="",
    )

    # Category of the model you want to test, default is "all"
    parser.add_argument(
        "--category",
        type=str,
        default="test_all",
        nargs="+",
        help="Category of the model you want to test",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Root directory of the test data.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        help="Root directory of the prediction data.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Root directory of the log data.",
    )

    # Temperature parameter to control randomness of model output, default is 0.7
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature parameter to control randomness of model output",
    )
    # Top-p parameter to control diversity of model output, default is 1
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p parameter to control diversity of model output",
    )
    # Maximum number of tokens to generate, default is 1200
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate",
    )

    # Language for model output, choose 'en' for English or 'zh' for Chinese
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Language for model output, choose 'en' for English or 'zh' for Chinese",
    )
    # Number of threads to use for concurrency, default is 1
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads to use for concurrency",
    )
    # Maximum number of dialog turns allowed for agent interactions
    parser.add_argument(
        "--max-dialog-turns",
        type=int,
        default=32,
        help="Maximum number of dialog turns allowed for agent interactions",
    )
    # Model used by the user role in the agent, it is recommended to use an advanced large model
    parser.add_argument(
        "--user-model-name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model used by the user role in the agent",
    )
    parser.add_argument(
        "--user-api-url",
        type=str,
        default="http://localhost:8181",
        help="",
    )
    parser.add_argument(
        "--user-api-key",
        type=str,
        default="EMPTY",
        help="",
    )

    args = parser.parse_args()
    return args


def generate_single_case(
    *,
    args,
    agent_model: BaseModelInference,
    test_case: Dict[str, Any],
    user_model: BaseModelInference | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    test_id = test_case["id"]

    # Multi-turn and multi-step mode.
    if "agent" in test_id:
        initial_config = test_case["initial_config"]
        involved_classes = test_case["involved_classes"]
        category = "multi_turn" if "multi_turn" in test_id else "multi_step"

        # Initialize agent model by injecting the system prompt.
        agent_system_prompt = compose_agent_system_prompt(
            category=category,
            involved_classes=involved_classes,
            language=args.language,
            model_name=args.model_name,
        )

        # Initialize user model by injecting the system prompt.
        if category == "multi_turn":
            user_message_history = compose_user_system_message(
                instruction=test_case["question"],
                involved_classes=involved_classes,
                language=args.language,
                model_name=args.user_model_name,
            )
        else:
            user_model = None
            user_message_history = None

        # Initialize the executor.
        executor = Executor(
            class_init_config=initial_config,
            involved_classes=involved_classes,
            language=args.language,
        )

        dialogue = inference(
            agent_model=agent_model,
            agent_system_prompt=agent_system_prompt,
            question=test_case["question"],
            functions=test_case["function"],
            max_dialog_turns=args.max_dialog_turns,
            generation_kwargs=generation_kwargs,
            executor=executor,
            user_model=user_model,
            user_message_history=user_message_history,
        )

        # Obtain `result` from the involved classes' status.
        class_status = executor.return_exe_class_status()

        # Obtain `process` from message (agent to executor) in the dialogue history.
        agent_fc_query_list = []
        for dia in dialogue:
            if dia["recipient"] == "executor":
                fc_call = dia["message"]["tool_calls"]
                fc_call_str = calls_to_pystr(fc_call)
                agent_fc_query_list.append(fc_call_str)

        result = {
            "id": test_id,
            "result": class_status,
            "process": agent_fc_query_list,
        }

    else:
        # Initialize agent model by injecting the system prompt.
        if "special" in test_id:
            agent_system_prompt = compose_special_system_prompt(
                time=test_case.get("time", ""),
                language=args.language,
                model_name=args.model_name,
            )
        elif "preferences" in test_id:
            agent_system_prompt = compose_preference_system_prompt(
                profile=test_case.get("profile", ""),
                language=args.language,
                model_name=args.model_name,
            )
        else:
            agent_system_prompt = compose_normal_system_prompt(
                time=test_case.get("time", ""),
                language=args.language,
                model_name=args.model_name,
            )

        # TODO: This is a Patch: convert the offline multi turn text into chat message format.
        # "user: what is the temperature?\nsystem: where?\nuser:in Paris."
        # [
        #   {"role": "user", "content": "what is the temperature?"},
        #   {"role": "assistant", "content": "where?"},
        #   {"role": "user", "content": "in Paris."},
        # ]
        if "multi_turn" in test_id:
            agent_message_history = convert_text_to_messages(test_case["question"])
            # The last one is always the user's query.
            question = agent_message_history.pop(-1)["content"]
        else:
            agent_message_history = None
            question = maybe_rm_role_in_text(test_case["question"])

        dialogue = inference(
            agent_model=agent_model,
            agent_system_prompt=agent_system_prompt,
            question=question,
            functions=test_case["function"],
            max_dialog_turns=1,
            generation_kwargs=generation_kwargs,
            agent_message_history=agent_message_history,
        )

        # Last dialogue should be from the agent to the executor (null).
        if dialogue[-1]["recipient"] == "executor":
            agent_fc = dialogue[-1]["message"]["tool_calls"]
            # Evaluation only allows a string like `[func1(arg1=val1),func2(arg2=val2)]`
            agent_fc_str = calls_to_pystr(agent_fc)
        else:
            warnings.warn(f"Last recipient is NOT executor, get {dialogue[-1]['recipient']}")
            agent_fc_str = dialogue[-1]["message"]["content"]

        result = {"id": test_id, "result": agent_fc_str}

    return result, dialogue


def generate_single_category(
    *,
    args,
    agent_model: BaseModelInference,
    file_path: str | os.PathLike,
    result_path: str | os.PathLike,
    log_path: str | os.PathLike,
    user_model: BaseModelInference | None = None,
    completed_id_set: set[str] | None = None,
):
    test_cases = []
    os.makedirs(Path(result_path).parent, exist_ok=True)
    os.makedirs(Path(log_path).parent, exist_ok=True)

    # Load cases (lines in the input file)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            test_cases.extend(json.loads(line) for line in file)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON in file - {file_path}")

    def has_completed(_test_case: Dict[str, Any]) -> bool:
        _test_id = _test_case["id"]
        if _test_id in completed_id_set:
            return True
        return False

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for test_case in test_cases:
            if has_completed(test_case):
                continue
            future = executor.submit(
                generate_single_case,
                args=args,
                agent_model=agent_model,
                test_case=test_case,
                user_model=user_model,
            )
            futures.append(future)

        # Collect results in original order
        results, logs = [None] * len(futures), [None] * len(futures)
        for idx, future in enumerate(futures):
            # .result() will re-raise exceptions from fun_call, surfacing errors early
            result, log = future.result()
            results[idx], logs[idx] = result, log

    # results = []
    # logs = []
    # for test_case in test_cases:
    #     if has_completed(test_case):
    #         continue
    #
    #     print(test_case["id"])
    #
    #     result, log = generate_single_case(
    #         args=args,
    #         agent_model=agent_model,
    #         test_case=test_case,
    #         user_model=user_model,
    #     )
    #
    #     results.append(result)
    #     logs.append(log)

    # Write the results and logs.
    print(f"===> Writing results to {result_path} ...\n")
    with open(result_path, "a", encoding="utf-8") as fp_result:
        with open(log_path, "a", encoding="utf-8") as fp_log:
            for result, log in zip(results, logs):
                fp_result.write(json.dumps(result, ensure_ascii=False) + "\n")
                fp_log.write(json.dumps(log, ensure_ascii=False) + "\n")


def main():
    args = parser()
    project_root = os.getenv("PROJECT_ROOT", "./")

    data_dir = Path(args.data_dir) if args.data_dir else Path(project_root) / "data"
    result_root_dir = Path(args.result_dir) if args.result_dir else Path(project_root) / "results"
    log_root_dir = Path(args.log_dir) if args.log_dir else Path(project_root) / "logs"

    # Get the filenames of the test cases
    test_names = {
        test_name for category in args.category for test_name in ACE_DATA_CATEGORY[category]
    }

    agent_inference = AGENT_NAME_MAP[args.model_name]
    agent_model = agent_inference(
        model_name=args.model_name,
        base_url=args.api_url,
        api_key=args.api_key,
    )

    user_model = UserModelInference(
        model_name=args.user_model_name,
        base_url=args.user_api_url,
        api_key=args.user_api_key,
    )

    for test_category in test_names:
        data_path = data_dir / args.language / f"data_{test_category}.json"
        result_path = (
            result_root_dir / args.model_name / args.language / f"result_{test_category}.json"
        )
        log_path = log_root_dir / args.model_name / args.language / f"log_{test_category}.json"

        # Count the cases that have already been generated to avoid duplication
        completed_id_set = set()
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_data = json.loads(line)
                    completed_id_set.add(line_data["id"])

        generate_single_category(
            args=args,
            agent_model=agent_model,
            file_path=data_path,
            result_path=result_path,
            log_path=log_path,
            user_model=user_model,
            completed_id_set=completed_id_set,
        )


if __name__ == "__main__":
    main()
