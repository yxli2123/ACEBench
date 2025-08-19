import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from category import ACE_DATA_CATEGORY
from data.data_utils import convert_text_to_messages
from model_inference.common_inference import inference, write_result
from model_inference.executor import Executor
from model_inference.prompt.prompt_utils import (
    compose_agent_system_prompt,
    compose_default_system_prompt,
    compose_preference_system_prompt,
    compose_special_system_prompt,
    compose_user_system_prompt,
)

# Directory of the ACEBench
PACKAGE_ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[0])


def parser():
    parser = argparse.ArgumentParser(description="Generate ACEBench Results.")
    # Model name
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3",
        help="Name of the model to use",
    )

    # For local models, specify the model path
    parser.add_argument(
        "--model-local-path",
        type=str,
        help="Path to the model for local models",
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
        "--eval-root-dir",
        type=str,
        help="Root directory of the evaluation data. It stores the generated result files, dialogue log files, and score files.",
    )

    # Temperature parameter to control randomness of model output, default is 0.7
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
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
        default=4000,
        help="Maximum number of tokens to generate",
    )
    # Number of GPUs to use, default is 1
    parser.add_argument(
        "--num-gpus", default=1, type=int, help="Number of GPUs to use"
    )
    # GPU memory utilization rate, default is 0.9
    parser.add_argument(
        "--gpu-memory-utilization",
        default=0.9,
        type=float,
        help="GPU memory utilization rate",
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
        default=40,
        help="Maximum number of dialog turns allowed for agent interactions",
    )
    # Model used by the user role in the agent, it is recommended to use an advanced large model
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model used by the user role in the agent",
    )
    parser.add_argument(
        "--enable-think",
        action="store_ture",
        help="Enable thinking in the system prompt.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size.",
    )
    parser.add_argument("--result-dir", type=str, help="result directory")

    args = parser.parse_args()
    return args


def load_test_cases(base_path, filenames):
    cases = []

    for filename in filenames:
        file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                cases.extend(json.loads(line) for line in file)
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse JSON in file - {file_path}")
    return cases


def sort_json(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if "multi_turn" in file and "agent" not in file:
        data = sorted(
            data, key=lambda x: tuple(map(int, x["id"].split("_")[-2:]))
        )
    else:
        data = sorted(data, key=lambda x: int(x["id"].split("_")[-1]))
    with open(file, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def generate_single_case(agent_model, test_case, user_model, args):
    result_path = args.result_path

    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    test_id = test_case["test_id"]

    # Multi-turn and multi-step mode.
    if "agent" in test_id:
        initial_config = test_case["initial_config"]
        involved_classes = test_case["involved_classes"]
        test_idx = test_id.split("_")[-1]
        category = "multi_turn" if "multi_turn" in test_id else "multi_step"

        # Initialize agent model by injecting the system prompt.
        agent_system_prompt = compose_agent_system_prompt(
            category, involved_classes, args.language
        )
        agent_model.inject_system_prompt(agent_system_prompt)

        # Initialize user model by injecting the system prompt.
        if "multi_turn" in test_case:
            user_system_prompt = compose_user_system_prompt()
            user_model.inject_system_prompt(user_system_prompt)
        else:
            user_model = None

        # Initialize the executor.
        executor = Executor(
            class_init_config=initial_config,
            involved_classes=involved_classes,
            language=args.language,
        )

        dialogue = inference(
            agent_model=agent_model,
            question=test_case["question"],
            functions=test_case["functions"],
            max_dialog_turns=16,
            generation_kwargs=generation_kwargs,
            executor=executor,
            user_model=user_model,
        )

        write_result(dialogue, result_path, mode="agent")

    else:
        # Initialize agent model by injecting the system prompt.
        if "special" in test_id:
            agent_system_prompt = compose_special_system_prompt()
        elif "preferences" in test_id:
            agent_system_prompt = compose_preference_system_prompt()
        else:
            agent_system_prompt = compose_default_system_prompt()

        agent_model.inject_system_prompt(agent_system_prompt)

        # Patch: convert the offline multi turn text into chat message format.
        if "multi_turn" in test_id:
            agent_message_history = convert_text_to_messages(
                test_case["question"]
            )
            # The last one is always the user's query.
            question = agent_message_history.pop(-1)["content"]
        else:
            agent_message_history = None
            question = test_case["question"]

        dialogue = inference(
            agent_model=agent_model,
            question=question,
            functions=test_case["functions"],
            max_dialog_turns=1,
            generation_kwargs=generation_kwargs,
            agent_message_history=agent_message_history,
        )

        write_result_function_text(dialogue, result_path)


def generate_results(args, model_name, test_case, completed_id_set):
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for test_case in test_cases_total:
            if test_case["id"] not in completed_id_set:
                future = executor.submit(
                    generate_single_case, model_name, test_case, args
                )
                futures.append(future)

        with tqdm(
            total=len(futures), desc="Processing Tasks", leave=True
        ) as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()  # Catch exceptions in tasks
                    pbar.update(1)
                except Exception as e:
                    print(f"Task raised an exception: {e}")
                    # You can choose whether to continue executing tasks after catching an exception, or to terminate the program
                    raise
        print("All tasks have been completed.")


def main():
    args = parser()

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.category) is not list:
        args.category = [args.category]

    project_root = os.getenv("PROJECT_ROOT", "./")

    paths = {
        "zh": {
            "data_path": "./data_all/data_zh/",
            "result_path": "./result_all/result_zh/",
        },
        "en": {
            "data_path": "./data_all/data_en/",
            "result_path": "./result_all/result_en/",
        },
    }

    data_path = paths[args.language]["data_path"]
    result_path = paths[args.language]["result_path"]
    args.result_path = result_path

    # Get the filenames of the test cases
    test_names = {
        test_name
        for category in args.category
        for test_name in ACE_DATA_CATEGORY[category]
    }
    test_files = [f"data_{test_name}.json" for test_name in test_names]

    for model_name in args.model:
        folder_path = os.path.join(result_path, model_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        completed_id_set = set()
        # Count the cases that have already been generated to avoid duplication
        for file in test_names:
            file_name = f"data_{file}_result.json"
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_data = json.loads(line)
                        completed_id_set.add(line_data["id"])
        # Read data
        test_cases_total = load_test_cases(data_path, test_files)

        if len(test_cases_total) > 0:
            generate_results(
                args, model_name, test_cases_total, completed_id_set
            )

        # Multithreading may disrupt the order of execution, so the result ids need to be reordered
        for file in test_names:
            file_name = f"data_{file}_result.json"
            file_path = os.path.join(folder_path, file_name)
            sort_json(file_path)


if __name__ == "__main__":
    main()
