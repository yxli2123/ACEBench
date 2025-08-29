import sys
from os import PathLike
from typing import List

sys.path.append("../")

import argparse

from category import ACE_DATA_CATEGORY
from model_eval.checker import *
from model_eval.evaluation_helper import *
from model_eval.utils import *
from model_inference.utils import decode_ast

RESULT_TABLE = {}


def normal_single_turn_eval(
    model_result, prompt, possible_answer, test_category, score_path
):
    if not all(len(x) == len(model_result) for x in [prompt, possible_answer]):
        raise ValueError(
            f"The length of the model result ({len(model_result)}) does not match "
            f"the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). "
            "Please check the input files for completeness."
        )

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        id = prompt[i]["id"]
        question = prompt[i]["question"]
        model_result_item = model_result[i]["result"]
        prompt_item = prompt[i]["function"]
        possible_answer_item = possible_answer[i]["ground_truth"]

        try:
            model_result_item_raw = model_result_item
            model_result_item_raw = "".join(model_result_item_raw.split())
            model_result_item = decode_ast(model_result_item_raw)
        except Exception as e:
            result.append(
                {
                    "id": id,
                    "valid": False,
                    "error": [
                        f"Invalid syntax. Failed to decode AST. {str(e)}"
                    ],
                    "error_type": "wrong_output_format",
                    "model_result_raw": model_result_item_raw,
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        # Check if the output format meets the requirements
        decoder_output_valid = is_function_call_format_valid(model_result_item)
        if not decoder_output_valid:
            result.append(
                {
                    "id": id,
                    "valid": False,
                    "error": [
                        "The output format does not meet the specified requirements."
                    ],
                    "error_type": "wrong_output_format",
                    "model_result": str(model_result_item_raw),
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        if type(possible_answer_item) != list:
            possible_answer_item = [possible_answer_item]

        all_errors = []

        # Filter from multiple candidate answers
        for possible_answer_item_ in possible_answer_item:
            checker_result = normal_checker(
                prompt_item,
                model_result_item,
                possible_answer_item_,
                question,
                test_category,
            )

            if checker_result["valid"]:
                correct_count += 1
                break
            else:
                all_errors.append(
                    {
                        "error": checker_result["error"],
                        "error_type": checker_result["error_type"],
                    }
                )

        if all_errors:
            temp = {
                "id": id,
                "valid": False,
                "error": all_errors[0]["error"],
                "error_type": all_errors[0]["error_type"],
                "model_result": model_result_item_raw,
                "possible_answer": possible_answer_item_,
            }
            result.append(temp)

    accuracy = round((correct_count / len(model_result)), 3)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    save_score_as_json(score_path, result)

    # convert_result_to_excel(model_name, test_category, paths)
    return accuracy


def normal_multi_turn_eval(
    model_result,
    prompt,
    possible_answer,
    test_category,
    score_path,
):
    if not all(len(x) == len(model_result) for x in [prompt, possible_answer]):
        raise ValueError(
            f"The length of the model result ({len(model_result)}) does not match "
            f"the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). "
            "Please check the input files for completeness."
        )

    result = []
    correct_count = 0

    process_score_list = []
    score_list = []

    for i in range(len(model_result)):
        id = model_result[i]["id"]
        turn = prompt[i]["id"].split("_")[-2]
        item = model_result[i]["id"].split("_")[-1]
        question = prompt[i]["question"]
        model_result_item = model_result[i]["result"]
        prompt_item = prompt[i]["function"]
        possible_answer_item_ = possible_answer[i]["ground_truth"]

        try:
            model_result_item_raw = model_result_item
            model_result_item_raw = "".join(model_result_item_raw.split())
            model_result_item = decode_ast(model_result_item_raw)
        except Exception as e:
            result.append(
                {
                    "id": id,
                    "turn": turn,
                    "valid": False,
                    "error": [
                        f"Invalid syntax. Failed to decode AST. {str(e)}"
                    ],
                    "error_type": "wrong_output_format",
                    "model_result": model_result_item_raw,
                    "possible_answer": possible_answer_item_,
                    "process": False,
                    "process_score": 0,
                }
            )

            process_score_list.append(0)

            if len(score_list) > 0 and turn == score_list[-1]["turn"]:
                score_list[-1]["valid"].append(False)
                score_list[-1]["number"] = item
            else:
                score_list.append(
                    {"turn": turn, "number": item, "valid": [False]}
                )
            continue
        # Check if the output format meets the requirements
        decoder_output_valid = is_function_call_format_valid(model_result_item)

        if not decoder_output_valid:
            result.append(
                {
                    "id": id,
                    "turn": turn,
                    "valid": False,
                    "error": [
                        "The output format does not meet the specified requirements."
                    ],
                    "error_type": "wrong_output_format",
                    "model_result": str(model_result_item),
                    "possible_answer": possible_answer_item_,
                    "process": False,
                    "process_score": 0,
                }
            )
            process_score_list.append(0)
            if len(score_list) > 0 and turn == score_list[-1]["turn"]:
                score_list[-1]["valid"].append(False)
                score_list[-1]["number"] = item
            else:
                score_list.append(
                    {"turn": turn, "number": item, "valid": [False]}
                )
            continue

        if type(possible_answer_item_) != list:
            possible_answer_item_ = [possible_answer_item_]

        all_errors = []

        # Filter from multiple candidate answers
        for possible_answer_item in possible_answer_item_:
            checker_result = normal_checker(
                prompt_item,
                model_result_item,
                possible_answer_item,
                question,
                test_category,
            )

            if checker_result["valid"]:
                correct_count += 1
                process_score_list.append(1)
                break
            else:
                all_errors.append(
                    {
                        "error": checker_result["error"],
                        "error_type": checker_result["error_type"],
                    }
                )

        if not checker_result["valid"]:
            temp = {
                "id": id,
                "turn": turn,
                "valid": False,
                "error": all_errors[0]["error"],
                "error_type": all_errors[0]["error_type"],
                "model_result": model_result_item_raw,
                "possible_answer": possible_answer_item_,
            }
            result.append(temp)

        turn = model_result[i]["id"].split("_")[-2]
        item = model_result[i]["id"].split("_")[-1]
        if len(score_list) > 0 and turn == score_list[-1]["turn"]:
            score_list[-1]["valid"].append(checker_result["valid"])
            score_list[-1]["number"] = item
        else:
            score_list.append(
                {
                    "turn": turn,
                    "number": item,
                    "valid": [checker_result["valid"]],
                }
            )

    if len(score_list) == 0:
        end_accuracy, process_accuracy = 0, 0
    else:
        end_accuracy, process_accuracy = multiplt_turn_accuracy(score_list)

    result.insert(
        0,
        {
            "accuracy": end_accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
            "process_accuracy": process_accuracy,
        },
    )

    save_score_as_json(score_path, result)

    # convert_result_to_excel(model_name, test_category, paths)

    return end_accuracy


def special_eval(
    model_result,
    prompt,
    possible_answer,
    category,
    score_path,
):
    if not all(len(x) == len(model_result) for x in [prompt, possible_answer]):
        raise ValueError(
            f"The length of the model result ({len(model_result)}) does not match "
            f"the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). "
            "Please check the input files for completeness."
        )

    result = []
    wrong_list = []
    correct_count = 0
    for i in range(len(model_result)):
        id = prompt[i]["id"]
        model_result_item = model_result[i]["result"]
        possible_answer_item_ = possible_answer[i]["ground_truth"]
        result.append(
            {
                "id": id,
                "valid": True,
                "error": [],
                "error_type": "",
                "model_result_decoded": str(model_result_item),
                "possible_answer": possible_answer_item_,
            }
        )
        if "incomplete" in category:
            for name, values in possible_answer_item_.items():
                if "Missing necessary parameters" not in model_result_item:
                    result[i]["valid"] = False
                    result[i]["error"] = [
                        f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"
                    ]
                    result[i]["error_type"] = "error_detection"
                elif name not in model_result_item:
                    result[i]["valid"] = False
                    result[i]["error"] = [
                        f"The user's instruction is missing necessary parameters ({values}) for the ({name}), but the model failed to correctly point it out"
                    ]
                    result[i]["error_type"] = "error_correction"
                else:
                    for value in values:
                        if value not in model_result_item:
                            result[i]["valid"] = False
                            result[i]["error"] = [
                                f"The user's instruction is missing necessary parameters ({value}) for the ({name}), but the model failed to correctly point it out"
                            ]
                            result[i]["error_type"] = "error_correction"

        elif "error" in category:
            for name, values in possible_answer_item_.items():
                if "There is incorrect value" not in model_result_item:
                    result[i]["valid"] = False
                    result[i]["error"] = [
                        f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"
                    ]
                    result[i]["error_type"] = "error_detection"
                else:
                    for value in values:
                        if value not in model_result_item:
                            result[i]["valid"] = False
                            result[i]["error"] = [
                                f"The user's instruction contains incorrect values ({values}) of the parameters ({name}), but the model failed to correctly point it out"
                            ]
                            result[i]["error_type"] = "error_correction"
        elif "irrelevant" in category:
            if "the limitations of the function" not in model_result_item:
                result[i]["valid"] = False
                result[i]["error"] = [
                    "The model cannot solve this problem, due to the limitations of the function"
                ]
                result[i]["error_type"] = "error_detection"

        if result[i]["valid"]:
            correct_count += 1

    for item in result:
        if item["valid"] == False:
            wrong_list.append(item)

    accuracy = correct_count / len(model_result)
    wrong_list.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    save_score_as_json(score_path, wrong_list)
    # convert_result_to_excel(model_name, category, paths)
    return accuracy


def agent_eval(
    model_result,
    prompt,
    possible_answer,
    test_category,
    score_path,
):
    if not all(len(x) == len(model_result) for x in [prompt, possible_answer]):
        raise ValueError(
            f"The length of the model result ({len(model_result)}) does not match "
            f"the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). "
            "Please check the input files for completeness."
        )

    result = []
    correct_index = []
    correct_count = 0

    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        possible_answer_item_ = possible_answer[i]["ground_truth"]

        if type(possible_answer_item_) != list:
            possible_answer_item_ = [possible_answer_item_]

        result_tmp = {
            "id": i,
            "valid": True,
            "error": [],
            "error_type": "",
        }

        is_valid = True
        checker_result = {}
        checker_result["valid"] = True

        if len(possible_answer_item_) != len(model_result_item):
            result_tmp["valid"] = False
            result_tmp["error_type"] = "wrong number of class"
            is_valid = False
        else:
            # Compare each category one by one
            for index in range(len(possible_answer_item_)):
                possible_keys = set(possible_answer_item_[index].keys())
                matched_dict = None
                for model_dict in model_result_item:
                    model_keys = set(model_dict.keys())
                    if possible_keys == model_keys:
                        matched_dict = model_dict
                        break

                if matched_dict:
                    checker_result = agent_checker(
                        matched_dict,
                        possible_answer_item_[index],
                    )

                if checker_result["valid"] == False:
                    result_tmp["valid"] = False
                    result_tmp["error"].append(checker_result["error"])
                    result_tmp["error_type"] = checker_result["error_type"]
                    is_valid = False

        if not is_valid:
            result.append(result_tmp)
        else:
            correct_count += 1
            correct_index.append(i)

    accuracy = round(correct_count / len(model_result), 3)
    process_accuracy, individual_accuracies = agent_eval_process(
        model_result,
        possible_answer,
        test_category,
        correct_index,
    )
    result.insert(
        0,
        {
            "end_to_end_accuracy": accuracy,
            "process_accuracy": process_accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )

    # Write individual_accuracies to JSON file line by line
    process_path = score_path.replace(".json", "_process.json")
    with open(process_path, "w", encoding="utf-8") as f:
        for entry in individual_accuracies:
            json.dump(entry, f, ensure_ascii=False)
            f.write(
                "\n"
            )  # Write a newline character to make each JSON object occupy a separate line
    save_score_as_json(score_path, result)
    return accuracy, process_accuracy


def agent_eval_process(
    model_results,
    possible_answers,
    test_category,
    correct_list,
):
    individual_accuracies = []  # Used to store the accuracy of each data point
    total_accuracy = 0  # Store the total accuracy of all data

    for index in range(len(model_results)):
        if index in correct_list:
            accuracy = 1.00
            total_accuracy += 1.00
            continue
        call_process = possible_answers[index]["mile_stone"]
        model_result = model_results[index]["process"]
        if isinstance(call_process[0], list):
            max_accuracy = -1
            for call_process_item in call_process:
                result_len = len(model_result)
                milestone_len = len(call_process_item)

                result_indices = []
                current_index = 0

                # Iterate through each element in call_process and search sequentially
                for stone in call_process_item:
                    # Start searching from the current index until the corresponding call_process element is found
                    while current_index < result_len:
                        if (
                            model_result[current_index].strip()
                            == stone.strip()
                        ):
                            result_indices.append(current_index)
                            current_index += 1
                            break
                        current_index += 1

                # Calculate call_process accuracy using floating-point division
                if milestone_len == 0:
                    accuracy = 1.00
                else:
                    accuracy = len(result_indices) / milestone_len
                rounded_accuracy = round(accuracy, 3)
                if rounded_accuracy > max_accuracy:
                    max_accuracy = rounded_accuracy
                    name = test_category + "_" + str(index)

            # Save the accuracy of each data point
            if accuracy != 1.00:
                individual_accuracies.append(
                    {
                        name: {
                            "process_accuracy": rounded_accuracy,
                            "model_output": model_result,
                            "call_process": call_process,
                        }
                    }
                )
            # Accumulate total accuracy
            total_accuracy += max_accuracy

        # For a single answer, calculate directly
        else:
            result_len = len(model_result)
            milestone_len = len(call_process)

            result_indices = []
            current_index = 0

            # Iterate through each element in call_process and search sequentially
            for stone in call_process:
                while current_index < result_len:
                    if model_result[current_index].strip() == stone.strip():
                        result_indices.append(current_index)
                        current_index += 1  # Update position and continue searching for the next stone
                        break
                    current_index += 1

            # Calculate call_process accuracy using floating-point division
            if milestone_len == 0:
                accuracy = 1.00
            else:
                accuracy = len(result_indices) / milestone_len
            rounded_accuracy = round(accuracy, 3)

            # Save the accuracy of each data point
            name = test_category + "_" + str(index)
            if accuracy != 1.00:
                individual_accuracies.append(
                    {
                        name: {
                            "process_accuracy": rounded_accuracy,
                            "model_output": model_result,
                            "call_process": call_process,
                        }
                    }
                )

            # Accumulate total accuracy
            total_accuracy += accuracy

    # Calculate the overall accuracy of all entries
    overall_accuracy = total_accuracy / len(model_results)
    overall_accuracy = round(overall_accuracy, 3)  # Keep two decimal places
    # if language == "zh":
    #     file_name = (
    #         "./score_all/score_zh/"
    #         + model_name
    #         + "/data_"
    #         + test_category
    #         + "_process.json"
    #     )
    # elif language == "en":
    #     file_name = (
    #         "./score_all/score_en/"
    #         + model_name
    #         + "/data_"
    #         + test_category
    #         + "_process.json"
    #     )

    # Return the accuracy of each data point and the overall accuracy
    return overall_accuracy, individual_accuracies


#### Main evaluate function ####
def evaluate(
    data_dir: PathLike,
    result_dir: PathLike,
    categories: List[str],
    score_dir: PathLike,
):
    for category in categories:
        print(f"🔍 Running test: {category}")

        score_path = os.path.join(score_dir, f"score_{category}.json")

        model_result_path = os.path.join(result_dir, f"result_{category}.json")
        model_result = load_file(model_result_path)

        prompt_path = os.path.join(data_dir, f"data_{category}.json")
        prompt = load_file(prompt_path)

        possible_answer_path = os.path.join(
            data_dir, "possible_answer", f"data_{category}.json"
        )
        possible_answer = load_file(possible_answer_path)

        if "special" in category:
            accuracy = special_eval(
                model_result,
                prompt,
                possible_answer,
                category,
                score_path,
            )
            print(f"✔️ Test '{category}' is done! 🚀 Accuracy: {accuracy}.")

        elif "agent" in category:
            end_accuracy, process_accuracy = agent_eval(
                model_result,
                prompt,
                possible_answer,
                category,
                score_path,
            )
            print(
                f"✔️ Test '{category}' is done! | End_to_End Accuracy: {end_accuracy} | Process Accuracy: {process_accuracy}"
            )

        elif "normal_multi_turn" in category:
            end_accuracy = normal_multi_turn_eval(
                model_result,
                prompt,
                possible_answer,
                category,
                score_path,
            )
            print(f"✔️ Test '{category}' is done! | Accuracy: {end_accuracy}")

        else:
            accuracy = normal_single_turn_eval(
                model_result,
                prompt,
                possible_answer,
                category,
                score_path,
            )
            print(f"✔️ Test '{category}' is done! | Accuracy: {accuracy}")

    update_result_table_with_score_file(RESULT_TABLE, OUTPUT_PATH)
    generate_result_csv(RESULT_TABLE, OUTPUT_PATH)


def get_paths(language):
    base_paths = {
        "zh": {
            "OUTPUT_PATH": "./score_all/score_zh/",
        },
        "en": {
            "OUTPUT_PATH": "./score_all/score_en/",
        },
    }
    return base_paths.get(language)


def main():
    parser = argparse.ArgumentParser(
        description="Process two lists of strings."
    )

    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument(
        "--model-name",
        type=str,
        help="A list of model names to evaluate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/",
        help="A list of model names to evaluate",
    )
    parser.add_argument(
        "--result-root-dir",
        type=str,
        default="./results/",
        help="A list of model names to evaluate",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        type=str,
        help="A list of test categories to run the evaluation on",
    )

    args = parser.parse_args()

    paths = get_paths(args.language)

    if paths:
        OUTPUT_PATH = paths["OUTPUT_PATH"]

    language = args.language

    # Extract test categories
    test_categories = [
        category
        for test_category in (args.category or [])
        for category in (ACE_DATA_CATEGORY.get(test_category, [test_category]))
    ]

    result_dir = os.path.join(
        args.result_root_dir, args.model_name, args.language
    )
    data_dir = os.path.join(args.data_dir, args.language)
    score_dir = os.path.join(
        args.score_root_dir, args.model_name, args.language
    )

    # Call the main function
    evaluate(
        data_dir=data_dir,
        result_dir=result_dir,
        categories=test_categories,
        score_dir=score_dir,
    )

    print(f"Models being evaluated: {args.model_name}")
    print(f"Test categories being used: {test_categories}")


if __name__ == "__main__":
    main()
