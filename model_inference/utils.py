import ast
import json
import re
import warnings
from typing import Any, Dict, List, Tuple


def decode_ast(result, language="Python"):
    decoded_output = ast_parse(result, language)
    return decoded_output


def ast_parse(input_str, language="Python"):
    if language == "Python":
        parsed = ast.parse(input_str, mode="eval")
        extracted = []
        for elem in parsed.body.elts:
            assert isinstance(elem, ast.Call)
            extracted.append(resolve_ast_by_type(elem))
        return extracted
    else:
        raise NotImplementedError(f"Unsupported language: {language}")


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v) for k, v in zip(value.keys, value.values)
        }
    elif isinstance(value, ast.NameConstant):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(value, ast.BinOp):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            # Handle no keyword arguments
            func_parts = []
            func_part = value.func
            while isinstance(func_part, ast.Attribute):
                func_parts.append(func_part.attr)
                func_part = func_part.value
            if isinstance(func_part, ast.Name):
                func_parts.append(func_part.id)
            func_name = ".".join(reversed(func_parts))
            output = {func_name: {}}
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except Exception:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


# ---------- Validators / helpers ----------

_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")
_DOTTED_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


def _is_valid_ident(name: str) -> bool:
    return bool(_IDENT_RE.match(name))


def _is_valid_dotted(name: str) -> bool:
    return bool(_DOTTED_RE.match(name))


def _ast_to_dotted_name(node: ast.AST) -> str:
    """
    Turn an AST of a function reference into a dotted string:
      Name(id='foo')                      -> "foo"
      Attribute(value=Name('a'), attr='b')-> "a.b"
      Attribute(value=Attribute(...), ...) -> "a.b.c"
    Reject anything else (e.g., subscripts, calls, lambdas).
    """
    parts: List[str] = []
    cur = node
    while True:
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            break
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
            continue
        raise ValueError("Only simple or dotted names are allowed for function calls.")
    parts.reverse()
    dotted = ".".join(parts)
    if not _is_valid_dotted(dotted):
        raise ValueError(f"Invalid function name: {dotted!r}")
    return dotted


def _literal_eval_node(node: ast.AST) -> Any:
    """ast.literal_eval with a clearer error."""
    try:
        return ast.literal_eval(node)
    except Exception as e:
        raise ValueError(
            "Argument values must be Python literals (e.g., numbers, strings, "
            "booleans, None, lists, dicts, tuples)."
        ) from e


def _value_to_literal_str(v: Any) -> str:
    """
    Convert a Python value into a Python-literal string (round-trippable by ast.literal_eval).
    For most cases, repr(...) is sufficient.
    """
    # Disallow NaN/Inf to keep round-trip safe via literal_eval
    if isinstance(v, float):
        if v != v:
            raise ValueError("NaN is not supported in literal output.")
        if v in (float("inf"), float("-inf")):
            raise ValueError("Infinity is not supported in literal output.")
    return repr(v)


# ---------- Public converters ----------


def calls_to_pystr(
    calls: List[Dict[str, Any]],
    *,
    sort_keys: bool = False,
) -> str:
    """
    Convert:
        [{"name": "f", "arguments": {"a": 1, "b": [2,3] }}, ...]
    into:
        "[f(a=1, b=[2, 3]), ...]"

    - Allows dotted function names (e.g., "pkg.mod.func").
    - Only keyword arguments are serialized (dict entries).
    - Values must be Python-literal friendly (round-trippable by ast.literal_eval).
    """
    if not isinstance(calls, list):
        raise TypeError("calls_to_pystr expects a list of call dicts.")

    call_strs: List[str] = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise TypeError(f"Call #{idx} must be a dict.")
        if "name" not in call or "arguments" not in call:
            raise ValueError(f"Call #{idx} must contain 'name' and 'arguments'.")

        name = call["name"]
        kwargs = call["arguments"]

        if not isinstance(name, str) or not _is_valid_dotted(name):
            raise ValueError(f"Invalid function name at index {idx}: {name!r}")

        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs)
            except Exception as e:
                warnings.warn(f"Failed to decode JSON str as dict.\n\n{e}\n\n{kwargs}")

        if not isinstance(kwargs, dict):
            raise TypeError(f"'arguments' for call #{idx} must be a dict, but got {type(kwargs)}")

        items: List[Tuple[str, Any]] = list(kwargs.items())
        if sort_keys:
            items = sorted(items, key=lambda kv: kv[0])

        # Validate keys are identifiers, and values are serializable literals
        kw_parts: List[str] = []
        for k, v in items:
            if not isinstance(k, str) or not _is_valid_ident(k):
                raise ValueError(f"Invalid argument name {k!r} in call to {name!r}.")
            kw_parts.append(f"{k}={_value_to_literal_str(v)}")

        call_strs.append(f"{name}({', '.join(kw_parts)})")

    return "[" + ", ".join(call_strs) + "]"


def pystr_to_calls(s: str) -> List[Dict[str, Any]]:
    """
    Convert:
        "[f(a=1, b=[2,3]), pkg.mod.g(x='hi')]"
    into:
        [{"name": "f", "arguments": {"a": 1, "b": [2, 3]}},
         {"name": "pkg.mod.g", "arguments": {"x": "hi"}}]

    Constraints:
    - Each element must be a function call with keyword args only.
    - No positional args, no *args/**kwargs, no calls as function objects.
    - Function name must be simple/dotted identifiers.
    - Argument values must be Python literals (ast.literal_eval).
    """
    if not isinstance(s, str):
        raise TypeError("pystr_to_calls expects a string.")

    s = s.strip()
    try:
        expr = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise ValueError("Input is not a valid Python expression of calls in a list.") from e

    if not isinstance(expr, ast.Expression) or not isinstance(expr.body, ast.List):
        raise ValueError("Top-level expression must be a Python list of function calls.")

    out: List[Dict[str, Any]] = []
    for i, elt in enumerate(expr.body.elts):
        if not isinstance(elt, ast.Call):
            raise ValueError(f"Element #{i} is not a function call.")

        # Disallow positional args
        if elt.args:
            raise ValueError(f"Call #{i} must use only keyword arguments (found positional args).")

        # Disallow **kwargs / *args
        for kw in elt.keywords:
            if kw.arg is None:
                raise ValueError(f"Call #{i} must not use **kwargs.")

        # Extract dotted function name
        func_name = _ast_to_dotted_name(elt.func)

        # Build arguments dict in written order
        args_dict: Dict[str, Any] = {}
        for kw in elt.keywords:
            if not _is_valid_ident(kw.arg):
                raise ValueError(f"Invalid argument name {kw.arg!r} in call #{i}.")
            args_dict[kw.arg] = _literal_eval_node(kw.value)

        out.append({"name": func_name, "arguments": args_dict})

    return out


def wrap_tool_protocol(
    tools: List[Dict[str, Any]],
    default_tool_type: str = "function",
) -> List[Dict[str, Any]]:
    """Convert [{"name": func_name, "...": ...}, ...] into
    [{"type": "function", "function": {"name": func_name, "...": ...}}, ...]"}]
    """
    wrapped_tools = []
    for tool in tools:
        wrapped_tools.append(
            {
                "type": default_tool_type,
                default_tool_type: tool,
            }
        )
    return wrapped_tools


def tool_output_to_message(
    tool_calls: List[Dict[str, Any]],
    tool_outputs: List[str],
) -> List[Dict[str, Any]]:
    """Example input:
    tool_calls = [
        {
            "name": "get_weather",
            "arguments": "...",
            "id": name-id(arguments)
        },
        {
            "name": "get_time",
            "arguments": "...",
            "id": name-id(arguments),
        },
    ]
    fun_output_list = [
        '{\"temp_c\": 22.0, \"conditions\": \"Sunny\"}',
        '\"2025-08-18 18:30:00 CEST\',
    ]
    Example output:
    [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": "{\"temp_c\": 22.0, \"conditions\": \"Sunny\"}"
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "get_time",
            "content": "\"2025-08-18 18:30:00 CEST\""
        },
    ]
    """
    tool_output_messages = []
    for tool_call, tool_output in zip(tool_calls, tool_outputs):
        tool_output_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_call["name"],
                "content": tool_output,
            }
        )

    return tool_output_messages
