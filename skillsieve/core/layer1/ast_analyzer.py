"""Layer 1 Module B: Tree-sitter AST feature extraction.

Parses script files using tree-sitter and extracts security-relevant
feature counts (syscalls, network calls, env access, eval/exec, encoded
strings) plus a Shannon entropy score for string literals.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser, Node

from skillsieve.models import SkillPackage

# ---------------------------------------------------------------------------
# Language setup
# ---------------------------------------------------------------------------

_LANGUAGES: dict[str, Language] = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
    "bash": Language(tsbash.language()),
}

_EXT_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}

# ---------------------------------------------------------------------------
# Python detection sets
# ---------------------------------------------------------------------------

_PY_EXEC_EVAL = {"eval", "exec"}

_PY_SYSCALL_ATTRS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"),
    ("os", "execve"),
    ("os", "execvp"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
}

_PY_NETWORK_ATTRS = {
    ("subprocess", "run"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "Popen"),
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "patch"),
    ("requests", "head"),
    ("requests", "request"),
    ("urllib", "request"),
    ("urllib", "urlopen"),
    ("http", "request"),
    ("httpx", "get"),
    ("httpx", "post"),
}

_PY_ENV_ATTRS = {
    ("os", "environ"),
    ("os", "getenv"),
}

# For subprocess calls, we check if the command contains network tools.
_NETWORK_CMDS = {"curl", "wget", "nc", "ncat", "netcat", "ssh", "scp"}

# Sensitive environment variable keywords (case-insensitive)
_SENSITIVE_ENV_KEYWORDS = {"key", "token", "secret", "password", "credential", "api"}

# Bash network commands
_BASH_NET_CMDS = {"curl", "wget", "nc", "ncat", "netcat"}

# Bash sensitive variable names (partial match)
_BASH_SENSITIVE_VAR_KEYWORDS = {"key", "token", "secret", "password", "credential", "api"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shannon_entropy(s: str) -> float:
    """Return Shannon entropy (bits) of a string."""
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum(
        (c / length) * math.log2(c / length)
        for c in counts.values()
        if c > 0
    )


def _get_attr_parts(node: Node) -> tuple[str, str] | None:
    """Return (object, attribute) for an attribute node, or None."""
    if node.type != "attribute":
        return None
    children = node.children
    # attribute nodes: <expr> '.' <identifier>
    if len(children) >= 3:
        obj_node = children[0]
        attr_node = children[-1]
        if attr_node.type == "identifier":
            obj_text = obj_node.text.decode("utf-8", errors="replace")
            attr_text = attr_node.text.decode("utf-8", errors="replace")
            return obj_text, attr_text
    return None


def _iter_nodes(root: Node):
    """Iterate all nodes in the tree using BFS."""
    queue = [root]
    while queue:
        node = queue.pop(0)
        yield node
        queue.extend(node.children)


# ---------------------------------------------------------------------------
# Language-specific extraction
# ---------------------------------------------------------------------------

def _extract_python(tree_root: Node) -> dict[str, float]:
    """Walk a Python AST and extract feature counts."""
    syscall_count = 0
    network_call_count = 0
    env_access_count = 0
    exec_eval_count = 0
    encoded_string_count = 0
    entropy_total = 0.0
    string_count = 0

    for node in _iter_nodes(tree_root):
        # --- eval / exec calls: identifier directly called ---
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is None and node.children:
                func_node = node.children[0]
            if func_node is not None:
                if func_node.type == "identifier":
                    name = func_node.text.decode("utf-8", errors="replace")
                    if name in _PY_EXEC_EVAL:
                        exec_eval_count += 1
                elif func_node.type == "attribute":
                    parts = _get_attr_parts(func_node)
                    if parts:
                        obj, attr = parts
                        # syscalls
                        if (obj, attr) in _PY_SYSCALL_ATTRS:
                            syscall_count += 1
                        # network / subprocess calls
                        if (obj, attr) in _PY_NETWORK_ATTRS:
                            # Check if this is subprocess with a network command
                            if obj == "subprocess":
                                # Look at first argument for network tools
                                args_node = node.child_by_field_name("arguments")
                                if args_node is None:
                                    for c in node.children:
                                        if c.type == "argument_list":
                                            args_node = c
                                            break
                                if args_node:
                                    cmd_text = args_node.text.decode("utf-8", errors="replace").lower()
                                    if any(nc in cmd_text for nc in _NETWORK_CMDS):
                                        network_call_count += 1
                                    else:
                                        network_call_count += 1  # subprocess itself is suspicious
                                else:
                                    network_call_count += 1
                            else:
                                network_call_count += 1
                        # env access: os.environ or os.getenv
                        if (obj, attr) in _PY_ENV_ATTRS:
                            env_access_count += 1
                # Also handle chained: os.environ.get(...) → inner is os.environ (attribute)
                elif func_node.type == "attribute":
                    pass  # handled above, but chained attrs need deeper check

        # --- os.environ attribute access (not necessarily a call) ---
        if node.type == "attribute":
            parts = _get_attr_parts(node)
            if parts and parts in _PY_ENV_ATTRS:
                # Only count if not already counted as a call
                parent = node.parent
                if parent is not None and parent.type != "call":
                    env_access_count += 1

        # --- String literals: entropy analysis ---
        if node.type == "string":
            raw = node.text.decode("utf-8", errors="replace")
            # Strip quotes for content analysis
            content = raw.strip("\"'bBrRuUfF")
            if len(content) >= 8:
                ent = _shannon_entropy(content)
                entropy_total += ent
                string_count += 1
                # High entropy strings (>= 4.5 bits) are suspicious encoded content
                if ent >= 4.5:
                    encoded_string_count += 1

    avg_entropy = entropy_total / string_count if string_count > 0 else 0.0

    return {
        "syscall_count": float(syscall_count),
        "network_call_count": float(network_call_count),
        "env_access_count": float(env_access_count),
        "exec_eval_count": float(exec_eval_count),
        "encoded_string_count": float(encoded_string_count),
        "entropy_score": avg_entropy,
    }


def _extract_javascript(tree_root: Node) -> dict[str, float]:
    """Walk a JavaScript/TypeScript AST and extract feature counts."""
    syscall_count = 0
    network_call_count = 0
    env_access_count = 0
    exec_eval_count = 0
    encoded_string_count = 0
    entropy_total = 0.0
    string_count = 0

    _JS_EXEC_EVAL = {"eval", "exec", "Function"}
    _JS_NET_METHODS = {"fetch", "get", "post", "put", "delete", "request", "open"}
    _JS_SYSCALL_FUNCS = {"execSync", "execFileSync", "spawnSync", "exec", "execFile", "spawn"}

    for node in _iter_nodes(tree_root):
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node is None and node.children:
                func_node = node.children[0]
            if func_node is not None:
                func_text = func_node.text.decode("utf-8", errors="replace")
                bare_name = func_text.split(".")[-1]

                if bare_name in _JS_EXEC_EVAL and "." not in func_text:
                    exec_eval_count += 1
                if bare_name in _JS_SYSCALL_FUNCS:
                    syscall_count += 1
                if bare_name in _JS_NET_METHODS or func_text == "fetch":
                    network_call_count += 1
                # process.env access
                if "process.env" in func_text:
                    env_access_count += 1

        # process.env member expressions
        if node.type == "member_expression":
            text = node.text.decode("utf-8", errors="replace")
            if text.startswith("process.env"):
                env_access_count += 1

        if node.type in ("string", "template_string"):
            raw = node.text.decode("utf-8", errors="replace").strip("`\"'")
            if len(raw) >= 8:
                ent = _shannon_entropy(raw)
                entropy_total += ent
                string_count += 1
                if ent >= 4.5:
                    encoded_string_count += 1

    avg_entropy = entropy_total / string_count if string_count > 0 else 0.0

    return {
        "syscall_count": float(syscall_count),
        "network_call_count": float(network_call_count),
        "env_access_count": float(env_access_count),
        "exec_eval_count": float(exec_eval_count),
        "encoded_string_count": float(encoded_string_count),
        "entropy_score": avg_entropy,
    }


def _extract_bash(tree_root: Node) -> dict[str, float]:
    """Walk a Bash AST and extract feature counts."""
    syscall_count = 0
    network_call_count = 0
    env_access_count = 0
    exec_eval_count = 0
    encoded_string_count = 0
    entropy_total = 0.0
    string_count = 0

    _BASH_SYSCALL_CMDS = {"chmod", "chown", "mount", "umount", "sudo", "su", "kill", "pkill"}
    _BASH_EXEC_CMDS = {"eval", "exec", "source"}

    for node in _iter_nodes(tree_root):
        if node.type == "command":
            cmd_name_node = None
            for c in node.children:
                if c.type == "command_name":
                    cmd_name_node = c
                    break
            if cmd_name_node:
                cmd = cmd_name_node.text.decode("utf-8", errors="replace").strip()
                if cmd in _BASH_NET_CMDS:
                    network_call_count += 1
                if cmd in _BASH_SYSCALL_CMDS:
                    syscall_count += 1
                if cmd in _BASH_EXEC_CMDS:
                    exec_eval_count += 1

        # Environment variable access via simple_expansion or variable_name
        if node.type in ("simple_expansion", "expansion"):
            var_node = None
            for c in node.children:
                if c.type == "variable_name":
                    var_node = c
                    break
            if var_node:
                var_name = var_node.text.decode("utf-8", errors="replace").lower()
                if any(kw in var_name for kw in _BASH_SENSITIVE_VAR_KEYWORDS):
                    env_access_count += 1

        # String entropy
        if node.type in ("string", "raw_string", "ansi_c_string"):
            raw = node.text.decode("utf-8", errors="replace").strip("\"'$")
            if len(raw) >= 8:
                ent = _shannon_entropy(raw)
                entropy_total += ent
                string_count += 1
                if ent >= 4.5:
                    encoded_string_count += 1

    avg_entropy = entropy_total / string_count if string_count > 0 else 0.0

    return {
        "syscall_count": float(syscall_count),
        "network_call_count": float(network_call_count),
        "env_access_count": float(env_access_count),
        "exec_eval_count": float(exec_eval_count),
        "encoded_string_count": float(encoded_string_count),
        "entropy_score": avg_entropy,
    }


_EXTRACTORS = {
    "python": _extract_python,
    "javascript": _extract_javascript,
    "bash": _extract_bash,
}

_ZERO_FEATURES: dict[str, float] = {
    "syscall_count": 0.0,
    "network_call_count": 0.0,
    "env_access_count": 0.0,
    "exec_eval_count": 0.0,
    "encoded_string_count": 0.0,
    "entropy_score": 0.0,
}


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ASTAnalyzer:
    """Extract AST-derived security features from skill script files."""

    def extract_features_from_file(self, code: str, language: str) -> dict[str, float]:
        """Parse *code* with tree-sitter and return a feature dict.

        Parameters
        ----------
        code:
            Source code as a plain string.
        language:
            One of ``"python"``, ``"javascript"``, or ``"bash"``.

        Returns
        -------
        dict with keys:
            syscall_count, network_call_count, env_access_count,
            exec_eval_count, encoded_string_count, entropy_score
        """
        lang_obj = _LANGUAGES.get(language)
        if lang_obj is None:
            return dict(_ZERO_FEATURES)

        parser = Parser(lang_obj)
        tree = parser.parse(code.encode("utf-8", errors="replace"))
        extractor = _EXTRACTORS.get(language)
        if extractor is None:
            return dict(_ZERO_FEATURES)

        return extractor(tree.root_node)

    def extract_features(self, pkg: SkillPackage) -> dict[str, float]:
        """Aggregate AST features across all script files in *pkg*.

        Returns a dict with the same keys as :meth:`extract_features_from_file`
        but prefixed with ``total_``.
        """
        totals: dict[str, float] = {
            "total_syscall_count": 0.0,
            "total_network_call_count": 0.0,
            "total_env_access_count": 0.0,
            "total_exec_eval_count": 0.0,
            "total_encoded_string_count": 0.0,
            "total_entropy_score": 0.0,
        }

        file_count = 0
        for file_path, content in pkg.script_files.items():
            suffix = "." + file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
            language = _EXT_MAP.get(suffix)
            if language is None:
                continue
            features = self.extract_features_from_file(content, language)
            file_count += 1
            totals["total_syscall_count"] += features["syscall_count"]
            totals["total_network_call_count"] += features["network_call_count"]
            totals["total_env_access_count"] += features["env_access_count"]
            totals["total_exec_eval_count"] += features["exec_eval_count"]
            totals["total_encoded_string_count"] += features["encoded_string_count"]
            totals["total_entropy_score"] += features["entropy_score"]

        # Average entropy across files
        if file_count > 0:
            totals["total_entropy_score"] /= file_count

        return totals
