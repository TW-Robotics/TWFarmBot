"""Optional restricted Python interpreter for registered FarmBot tools.

This is not the planner path. The assistant and ``plan()`` use JSON
function/tool calling. Hardware is never touched except through the same
``ApprovalGate`` / ``ActionRegistry`` path used by JSON tool calls.
"""

from __future__ import annotations

import ast
import io
import json
import re
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Union, get_args, get_origin

from pydantic import ValidationError

from .tool_policy import ToolDescriptor


_FENCE_RE = re.compile(r"```(?:python|py)[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

_MAX_SCRIPT_CHARS = 20_000
_MAX_STEPS = 10_000
_MAX_TOOL_CALLS = 200
_MAX_RESULT_CHARS = 8_000

_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "quit",
        "setattr",
        "vars",
        "dir",
        "memoryview",
        "help",
        "copyright",
        "credits",
        "license",
    }
)

_ALLOWED_NODES = frozenset(
    {
        ast.Module,
        ast.Expression,
        ast.Interactive,
        ast.FunctionDef,
        ast.Return,
        ast.Assign,
        ast.AnnAssign,
        ast.AugAssign,
        ast.For,
        ast.While,
        ast.If,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.Expr,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.IfExp,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.keyword,
        ast.arguments,
        ast.arg,
        ast.Lambda,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Starred,
        ast.NamedExpr,
        ast.Try,
        ast.ExceptHandler,
        ast.Raise,
        ast.Assert,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.LShift,
        ast.RShift,
        ast.Invert,
        ast.MatMult,
    }
)
if hasattr(ast, "Index"):
    _ALLOWED_NODES = frozenset(
        _ALLOWED_NODES | {ast.Index, getattr(ast, "ExtSlice", ast.Slice)}
    )


class FarmScriptError(ValueError):
    """Raised when a farm script is rejected or aborts."""


@dataclass
class ScriptResult:
    """Outcome of one farm-script execution."""

    ok: bool
    calls: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    result: Any = None
    error: str = ""


def extract_farm_scripts(
    text: str, tool_names: Iterable[str] | None = None
) -> list[str]:
    """Return Python sources the model asked to run.

    Prefers fenced python blocks. If none are present and the whole
    message parses as a script that calls a known tool, that message is
    treated as a script (models often omit the language tag).
    """
    if not text:
        return []
    fenced = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    fenced = [src for src in fenced if src]
    if fenced:
        return fenced

    names = set(tool_names or ())
    stripped = text.strip()
    if names and _looks_like_script(stripped, names):
        return [stripped]
    return []


def format_tool_catalog(descriptors: Iterable[ToolDescriptor]) -> str:
    """Render tools as signatures for the system prompt."""
    lines = [
        "Call the tools below with JSON function/tool calling.",
        "Do not write Python scripts or fenced farm_script blocks.",
        "The runtime executes each call and returns a JSON result.",
        "Physical ACT tools go through approval and safety before motors move.",
        "",
        "Available tools:",
    ]
    for descriptor in descriptors:
        lines.append(f"- `{_signature(descriptor)}`")
        extra = descriptor.policy.description.strip()
        approval = ""
        if descriptor.policy.requires_approval:
            approval = (
                " Requires user approval (becomes a proposal unless already allowed)."
            )
        elif descriptor.policy.category.value == "act":
            approval = " Executes immediately when actions are allowed."
        if extra or approval:
            lines.append(f"  {extra}{approval}".rstrip())
    return "\n".join(lines)


def format_script_feedback(result: ScriptResult) -> str:
    """Compact result payload to send back to the model."""
    payload: dict[str, Any] = {
        "ok": result.ok,
        "calls": [
            {
                "name": call["name"],
                "args": call.get("args", {}),
                "result": _llm_trim(call.get("result")),
            }
            for call in result.calls
        ],
    }
    if result.stdout:
        payload["stdout"] = result.stdout[-_MAX_RESULT_CHARS:]
    if result.result is not None:
        payload["result"] = _llm_trim(result.result)
    if result.error:
        payload["error"] = result.error
    encoded = json.dumps(payload, default=str, ensure_ascii=True)
    if len(encoded) > _MAX_RESULT_CHARS:
        encoded = encoded[:_MAX_RESULT_CHARS] + "...(truncated)"
    return "Farm script results:\n" + encoded


class FarmScriptRuntime:
    """Execute model-written Python against registered tools."""

    def __init__(
        self,
        descriptors: Iterable[ToolDescriptor],
        invoke: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self._descriptors = {d.name: d for d in descriptors}
        self._invoke = invoke
        self._calls: list[dict[str, Any]] = []
        self._steps = 0
        self._namespace: dict[str, Any] = self._fresh_namespace()

    def reset(self) -> None:
        self._calls = []
        self._steps = 0
        self._namespace = self._fresh_namespace()

    def run(self, source: str) -> ScriptResult:
        self._calls = []
        self._steps = 0
        if len(source) > _MAX_SCRIPT_CHARS:
            return ScriptResult(
                ok=False, error=f"script longer than {_MAX_SCRIPT_CHARS} characters"
            )
        try:
            tree = ast.parse(source, filename="<farm_script>", mode="exec")
            _validate_ast(tree)
            tree = _prepare_tree(tree)
        except (SyntaxError, FarmScriptError) as err:
            return ScriptResult(ok=False, error=str(err))

        stdout = io.StringIO()

        def _print(*args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("file", stdout)
            print(*args, **kwargs)

        self._namespace["print"] = _print
        self._namespace["_guard"] = self._guard
        self._namespace["_result"] = None

        try:
            compiled = compile(tree, "<farm_script>", "exec")
            exec(compiled, self._namespace, self._namespace)  # noqa: S102
        except FarmScriptError as err:
            return ScriptResult(
                ok=False,
                calls=list(self._calls),
                stdout=stdout.getvalue(),
                error=str(err),
            )
        except Exception as err:  # noqa: BLE001
            return ScriptResult(
                ok=False,
                calls=list(self._calls),
                stdout=stdout.getvalue(),
                error=f"{type(err).__name__}: {err}",
            )

        return ScriptResult(
            ok=True,
            calls=list(self._calls),
            stdout=stdout.getvalue(),
            result=self._namespace.get("_result"),
        )

    def _fresh_namespace(self) -> dict[str, Any]:
        builtins: dict[str, Any] = {
            "True": True,
            "False": False,
            "None": None,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "RuntimeError": RuntimeError,
        }
        ns: dict[str, Any] = {"__builtins__": builtins, "__name__": "farm_script"}
        for name, descriptor in self._descriptors.items():
            ns[name] = self._wrap_tool(descriptor)
        return ns

    def _wrap_tool(self, descriptor: ToolDescriptor) -> Callable[..., Any]:
        schema = descriptor.args_schema
        field_names = list(schema.model_fields)

        def _tool(*args: Any, **kwargs: Any) -> Any:
            if len(self._calls) >= _MAX_TOOL_CALLS:
                raise FarmScriptError(f"script exceeded {_MAX_TOOL_CALLS} tool calls")
            merged = dict(kwargs)
            if args:
                if len(args) > len(field_names):
                    raise FarmScriptError(
                        f"{descriptor.name}() got {len(args)} positional arguments"
                    )
                for key, value in zip(field_names, args):
                    if key in merged:
                        raise FarmScriptError(
                            f"{descriptor.name}() got multiple values for {key!r}"
                        )
                    merged[key] = value
            try:
                parsed = schema(**merged)
            except ValidationError as err:
                raise FarmScriptError(
                    f"invalid arguments for {descriptor.name}(): {err}"
                ) from err
            params = parsed.model_dump()
            result = self._invoke(descriptor.name, params)
            self._calls.append(
                {"name": descriptor.name, "args": params, "result": result}
            )
            return result

        _tool.__name__ = descriptor.name
        _tool.__doc__ = descriptor.policy.description
        return _tool

    def _guard(self) -> None:
        self._steps += 1
        if self._steps > _MAX_STEPS:
            raise FarmScriptError(f"script exceeded {_MAX_STEPS} steps")


def _looks_like_script(source: str, tool_names: set[str]) -> bool:
    if source.startswith("{") or source.startswith("["):
        return False
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in tool_names:
                return True
    return False


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise FarmScriptError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (
            node.id in _FORBIDDEN_NAMES or node.id.startswith("_")
        ):
            raise FarmScriptError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
            raise FarmScriptError(f"name {node.name!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise FarmScriptError("private attribute access is not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_NAMES:
                raise FarmScriptError(f"call to {node.func.id!r} is not allowed")


def _prepare_tree(tree: ast.Module) -> ast.Module:
    tree = _GuardTransformer().visit(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1] = ast.Assign(
            targets=[ast.Name(id="_result", ctx=ast.Store())],
            value=tree.body[-1].value,
        )
    else:
        tree.body.append(
            ast.Assign(
                targets=[ast.Name(id="_result", ctx=ast.Store())],
                value=ast.Constant(value=None),
            )
        )
    return ast.fix_missing_locations(tree)


class _GuardTransformer(ast.NodeTransformer):
    def _guard_stmt(self) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="_guard", ctx=ast.Load()), args=[], keywords=[]
            )
        )

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body = [self._guard_stmt(), *node.body]
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body = [self._guard_stmt(), *node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = [self._guard_stmt(), *node.body]
        return node


def _signature(descriptor: ToolDescriptor) -> str:
    fields = descriptor.args_schema.model_fields
    parts: list[str] = []
    for name, field_info in fields.items():
        anno = _type_name(field_info.annotation)
        if field_info.is_required():
            parts.append(f"{name}: {anno}")
        else:
            default = field_info.default
            if default is None or isinstance(default, (str, int, float, bool)):
                parts.append(f"{name}: {anno} = {default!r}")
            else:
                parts.append(f"{name}: {anno} = ...")
    return f"def {descriptor.name}({', '.join(parts)}) -> dict"


def _type_name(annotation: Any) -> str:
    if annotation is None:
        return "Any"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if annotation is type(None):
            return "None"
        if isinstance(annotation, type):
            return annotation.__name__
        return "Any"
    origin_name = getattr(origin, "__name__", None) or str(origin)
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(arg) for arg in args) or "Any"
    if args:
        inner = ", ".join(_type_name(arg) for arg in args)
        simple = {"list": "list", "dict": "dict", "tuple": "tuple", "set": "set"}
        return f"{simple.get(origin_name, origin_name)}[{inner}]"
    return origin_name


def _llm_trim(value: Any) -> Any:
    if isinstance(value, dict) and "image_url" in value:
        out = dict(value)
        out["image_url"] = "[image data shown to user in chat]"
        return out
    return value
