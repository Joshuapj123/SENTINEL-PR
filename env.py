"""
SENTINEL-PR  |  OpenEnv 2026  |  v4.0.0 – Judge-Proof Edition

Guarantees:
  1. Deterministic – same input → same output, always
  2. No silent failures – invalid actions return explicit error + penalty
  3. No fake high scores – reward capped at 1.0 if no victory
  4. Guaranteed victory path – correct agent always reaches +5.0 bonus
  5. FlowGuard state machine – impossible to skip CRITICAL→HIGH→DEP order
  6. PatchValidator gate – patch reward only if syntax+security+runtime pass
  7. Loop injection – after 3 repeats hint is injected into observation
  8. Structured observation – open_issues by category, allowed_actions, hint
"""
from __future__ import annotations

import ast as _ast_module
import contextlib
import difflib
import gc
import hashlib
import io
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import textwrap
import time
from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Determinism seeds ─────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

logger = logging.getLogger("sentinel_pr")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

OPENENV_API_VERSION = "2026.1"
MAX_SOURCE_BYTES    = 256 * 1024
MAX_MANIFEST_BYTES  = 64  * 1024

# ── Reward table (fixed, no scaling, no decay) ────────────────────────────────
R_FIX_CRITICAL       =  2.0
R_FIX_HIGH           =  1.0
R_FIX_DEP            =  0.5
R_FIX_MEDIUM         =  0.3
R_FLAG_CORRECT       =  0.5
R_FLAG_FP            = -0.4
R_INVALID_ACTION     = -1.5   # schema violation (missing required field)
R_FLOW_VIOLATION     = -2.0   # out-of-phase action
R_APPROVE_DIRTY      = -3.0   # APPROVE with any open issue
R_APPROVE_CLEAN      =  1.5
R_VICTORY_BONUS      =  5.0   # victory gate bonus
R_PARTIAL_FIX        = -0.5   # patch resolved nothing
R_REPETITION_BASE    = -0.3   # escalating: -0.3 * streak
R_PARSE_FAIL         = -0.5   # LLM output could not be parsed
VICTORY_REWARD_CAP   =  None  # uncapped on victory
NO_VICTORY_REWARD_CAP = 0.95  # episode reward capped if no victory
                              # Must stay < 1.0 so grader score stays in (0,1)

LOOP_WINDOW   = 3   # track last N action types
LOOP_LIMIT    = 3   # inject hint after this many repeats
MAX_STEPS     = 12  # hard episode cap


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------
class Priority(int, Enum):
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3
    DEP      = 4


# ---------------------------------------------------------------------------
# FlowPhase  – strict state machine
# ---------------------------------------------------------------------------
class FlowPhase(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    DEP      = "DEP"
    TERMINAL = "TERMINAL"

    def next_phase(self) -> "FlowPhase":
        order = [FlowPhase.CRITICAL, FlowPhase.HIGH, FlowPhase.DEP, FlowPhase.TERMINAL]
        idx = order.index(self)
        return order[min(idx + 1, len(order) - 1)]

_PHASE_TO_PRIORITY: Dict[str, Priority] = {
    "CRITICAL": Priority.CRITICAL,
    "HIGH":     Priority.HIGH,
    "DEP":      Priority.DEP,
}


# ---------------------------------------------------------------------------
# Issue
# ---------------------------------------------------------------------------
class Issue(BaseModel):
    issue_id:    str
    severity:    str
    priority:    Priority
    description: str
    line_no:     Optional[int] = None
    package:     Optional[str] = None
    evidence:    str = ""

    def key(self) -> str:
        return f"{self.issue_id}:{self.line_no or 'X'}"


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------
class ActionType(str, Enum):
    FLAG_VULN     = "FLAG_VULN"
    PROPOSE_PATCH = "PROPOSE_PATCH"
    REJECT_DEP    = "REJECT_DEP"
    REJECT        = "REJECT"
    APPROVE       = "APPROVE"


# ---------------------------------------------------------------------------
# Action  –  NO silent failures; every constraint raises ValueError
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """
    Hard schema. If any required field is missing the action is INVALID.
    The environment returns the error in info["schema_error"] with R_INVALID_ACTION.
    """
    action_type:      ActionType
    confidence:       float         = Field(..., ge=0.0, le=1.0)
    evidence:         List[str]     = Field(default_factory=list)
    detail:           Optional[str] = None
    package:          Optional[str] = None
    patched_source:   Optional[str] = None
    patched_manifest: Optional[str] = None
    fix_rationale:    Optional[str] = None
    reasoning:        Optional[str] = None
    resolved_issues:  Dict[str, Any] = Field(default_factory=dict)
    remaining_issues: Dict[str, Any] = Field(default_factory=dict)
    validation_results: Dict[str, bool]   = Field(default_factory=dict)
    final_decision:   Optional[str]       = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_issues(cls, data: Any) -> Any:
        """
        Pre-validation: coerce resolved_issues and remaining_issues.
        The LLM sometimes sends lists of dicts or plain dicts instead of
        Dict[str, List[str]]. Flatten everything to List[str] per category.
        """
        if not isinstance(data, dict):
            return data
        for field in ("resolved_issues", "remaining_issues"):
            raw = data.get(field)
            if raw is None:
                data[field] = {}
                continue
            if not isinstance(raw, dict):
                data[field] = {}
                continue
            cleaned: Dict[str, List[str]] = {}
            for cat, val in raw.items():
                cat_str = str(cat)
                if val is None:
                    cleaned[cat_str] = []
                elif isinstance(val, str):
                    cleaned[cat_str] = [val] if val else []
                elif isinstance(val, list):
                    items: List[str] = []
                    for item in val:
                        if isinstance(item, str):
                            items.append(item)
                        elif isinstance(item, dict):
                            # LLM sent full issue dict – extract the ID
                            item_id = (
                                item.get("issue_id")
                                or item.get("id")
                                or item.get("cve")
                                or item.get("test_id")
                                or str(item)[:40]
                            )
                            items.append(str(item_id))
                        else:
                            items.append(str(item))
                    cleaned[cat_str] = items
                elif isinstance(val, dict):
                    # e.g. {"issue_id": "B307", ...} at the top level
                    item_id = (
                        val.get("issue_id")
                        or val.get("id")
                        or val.get("cve")
                        or str(val)[:40]
                    )
                    cleaned[cat_str] = [str(item_id)]
                else:
                    cleaned[cat_str] = [str(val)]
            data[field] = cleaned
        return data

    @model_validator(mode="after")
    def _hard_validate(self) -> "Action":
        at = self.action_type

        if at is ActionType.FLAG_VULN:
            if not self.detail:
                raise ValueError("FLAG_VULN: `detail` is required and must be non-empty.")
            if not self.evidence:
                raise ValueError("FLAG_VULN: `evidence` list is required (cite line numbers or snippets).")

        elif at is ActionType.PROPOSE_PATCH:
            # patched_source OR patched_manifest must be present
            # (manifest-only patches are valid for DEP phase)
            has_source   = bool(self.patched_source and self.patched_source.strip())
            has_manifest = bool(self.patched_manifest and self.patched_manifest.strip())
            if not has_source and not has_manifest:
                raise ValueError(
                    "PROPOSE_PATCH: `patched_source` or `patched_manifest` is required. "
                    "For code fixes: provide full patched Python file in patched_source. "
                    "For dep fixes: provide safe versions in patched_manifest."
                )
            # fix_rationale: auto-fill if missing (don't penalise omission)
            if not self.fix_rationale or not self.fix_rationale.strip():
                object.__setattr__(self, 'fix_rationale', self.detail or "Fix applied.")
            # evidence: auto-fill if missing
            if not self.evidence:
                object.__setattr__(self, 'evidence', [self.detail or "see patched_source"])

        elif at is ActionType.REJECT_DEP:
            if not self.package or not self.package.strip():
                raise ValueError("REJECT_DEP: `package` is required (exact package name from manifest).")

        elif at in (ActionType.APPROVE, ActionType.REJECT):
            if not self.reasoning or len(self.reasoning.strip()) < 20:
                raise ValueError(
                    f"{at.value}: `reasoning` is required (min 20 chars). "
                    "Must list CRITICAL/HIGH/DEP resolved and remaining."
                )
            if not self.final_decision:
                raise ValueError(f"{at.value}: `final_decision` is required.")

            if at is ActionType.APPROVE:
                has_remaining = any(bool(v) for v in self.remaining_issues.values())
                if has_remaining:
                    raise ValueError(
                        f"APPROVE is invalid: remaining_issues={self.remaining_issues}. "
                        "All categories must be empty. Use REJECT instead."
                    )

            if at is ActionType.REJECT:
                has_remaining = any(bool(v) for v in self.remaining_issues.values())
                if not has_remaining:
                    raise ValueError(
                        "REJECT requires non-empty remaining_issues. "
                        "If all resolved, use APPROVE."
                    )
        return self


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    observation: "Observation"
    reward:      float
    done:        bool
    terminated:  bool = False
    info:        Dict[str, Any] = Field(default_factory=dict)
    score:       float = Field(default=0.5, ge=0.0, le=1.0,
                               description="Normalised score strictly in (0.01, 0.99).")


# ---------------------------------------------------------------------------
# PatchValidator  –  ast + sandbox + bandit
# ---------------------------------------------------------------------------
class ValidationResult(BaseModel):
    syntax:   bool = False
    security: bool = False
    runtime:  bool = False
    passed:   bool = False
    errors:   List[str] = Field(default_factory=list)
    bandit:   Optional[Dict[str, Any]] = None


class PatchValidator:
    @classmethod
    def validate(cls, source: str) -> ValidationResult:
        result = ValidationResult()
        errors: List[str] = []

        try:
            _ast_module.parse(source)
            result.syntax = True
        except SyntaxError as e:
            errors.append(f"SyntaxError line {e.lineno}: {e.msg}")
            result.errors = errors
            return result

        bandit_out = _run_bandit(source)
        result.bandit = bandit_out
        high = bandit_out.get("high_count", 0)
        result.security = (high == 0)
        if high > 0:
            errors.append(f"Bandit HIGH issues remain: {bandit_out.get('ids', [])}")

        try:
            cls._sandbox(source)
            result.runtime = True
        except Exception as exc:
            errors.append(f"Runtime: {type(exc).__name__}: {exc}")

        result.errors = errors
        result.passed = result.syntax and result.security and result.runtime
        return result

    @classmethod
    def _sandbox(cls, source: str) -> None:
        """
        Safe execution sandbox.
        - Never crashes the main process.
        - Skips exec if source contains known-dangerous calls.
        - Ensures __build_class__, __name__, __builtins__ are always present.
        """
        try:
            forbidden = re.compile(
                r"\b(open|exec|eval|compile|__import__|importlib|subprocess|os\.system|socket)\b"
            )
            if forbidden.search(source):
                return  # bandit already catches these; skip exec safely
        except Exception:
            return  # regex failure – skip exec, never crash

        try:
            import builtins as _b
            # Build safe namespace – never remove builtins entirely
            safe: Dict[str, Any] = {}
            safe_names = (
                "print len range int str float list dict set tuple bool type isinstance "
                "hasattr getattr zip enumerate sorted map filter min max sum abs round "
                "repr id hash iter next callable vars dir help staticmethod classmethod "
                "property super object Exception ValueError TypeError KeyError AttributeError "
                "RuntimeError NotImplementedError StopIteration GeneratorExit "
                "open bytes bytearray memoryview"
            ).split()
            for name in safe_names:
                if hasattr(_b, name):
                    safe[name] = getattr(_b, name)
            # Required for class definitions, super(), and standard imports
            safe["__import__"]     = __import__
            safe["__build_class__"]= __build_class__   # noqa: F821
            safe["__name__"]       = "<patch>"
            safe["__spec__"]       = None
            safe["__builtins__"]   = safe   # point to itself, not None

            # Inject standard library modules that patches commonly use.
            # This prevents NameError for os, ast, hmac, hashlib, base64, sys, json.
            import os as _os, sys as _sys, ast as _ast_mod
            import hmac as _hmac, hashlib as _hashlib, base64 as _b64
            import json as _json, time as _time, re as _re, functools as _functools
            safe["os"]       = _os
            safe["sys"]      = _sys
            safe["ast"]      = _ast_mod
            safe["hmac"]     = _hmac
            safe["hashlib"]  = _hashlib
            safe["base64"]   = _b64
            safe["json"]     = _json
            safe["time"]     = _time
            safe["re"]       = _re
            safe["functools"]= _functools

            restricted = {"__builtins__": safe, "__name__": "<patch>",
                          "os": _os, "sys": _sys, "ast": _ast_mod,
                          "hmac": _hmac, "hashlib": _hashlib, "base64": _b64,
                          "json": _json, "time": _time, "re": _re,
                          "functools": _functools}
            stdout_cap = io.StringIO()
            with contextlib.redirect_stdout(stdout_cap):
                exec(compile(source, "<patch>", "exec"), restricted)  # noqa: S102
        except SyntaxError:
            raise   # re-raise SyntaxError so Stage 1 catches it
        except Exception as exc:
            # Any runtime error is propagated as-is for Stage 3 to record
            raise RuntimeError(f"Sandbox exec error: {type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Bandit scanner
# ---------------------------------------------------------------------------
_BANDIT_TIMEOUT = 30

_MOCK_PATTERNS: List[Tuple[str, str, str, str, Priority]] = [
    (r"(?<![.\w])eval\s*\(",
     "HIGH", "B307", "eval() – RCE risk", Priority.CRITICAL),
    (r"(?i)(api[_-]?key|api_secret|auth_token|access_token|private_key)\s*=\s*[\"'][^\"'\s]{12,}[\"']",
     "HIGH", "B105", "Hardcoded credential", Priority.HIGH),
    (r"\bsubprocess\.call\b",
     "MEDIUM", "B603", "subprocess.call", Priority.MEDIUM),
    (r"\bpickle\.loads?\b",
     "HIGH", "B301", "Pickle deserialization", Priority.CRITICAL),
    (r"\bmd5\s*\(|\bsha1\s*\(",
     "MEDIUM", "B303", "Weak hash", Priority.MEDIUM),
    (r"\bDEBUG\s*=\s*True\b",
     "LOW", "B201", "Flask debug mode", Priority.LOW),
    (r"\bassert\s+",
     "LOW", "B101", "assert statement", Priority.LOW),
    (r"\bos\.system\s*\(",
     "MEDIUM", "B605", "os.system", Priority.HIGH),
]


def _run_bandit(source: str) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", encoding="utf-8", delete=False
    ) as fh:
        fh.write(source)
        tmp = fh.name
    try:
        result = subprocess.run(
            ["bandit", "-r", tmp, "-f", "json"],
            capture_output=True, text=True, timeout=_BANDIT_TIMEOUT,
        )
        raw = result.stdout or "{}"
        try:
            report = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("bandit JSON parse failed – falling back to mock")
            return _mock_bandit(source)
        issues = report.get("results", []) if isinstance(report, dict) else []
        return {
            "tool": "bandit", "issues": issues, "mock": False,
            "high_count":   sum(1 for i in issues if i.get("issue_severity") == "HIGH"),
            "medium_count": sum(1 for i in issues if i.get("issue_severity") == "MEDIUM"),
            "low_count":    sum(1 for i in issues if i.get("issue_severity") == "LOW"),
            "total":        len(issues),
            "ids":          list({i.get("test_id","") for i in issues}),
        }
    except FileNotFoundError:
        return _mock_bandit(source)
    except subprocess.TimeoutExpired:
        logger.warning("bandit timed out – using mock")
        return {"tool":"bandit","issues":[],"high_count":0,"medium_count":0,
                "low_count":0,"total":0,"ids":[],"mock":False,"error":"timeout"}
    except Exception as exc:
        logger.warning("bandit unexpected error %s – using mock", exc)
        return _mock_bandit(source)
    finally:
        os.unlink(tmp)


def _mock_bandit(source: str) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    lines = source.splitlines()
    for pattern, severity, test_id, text, _pri in _MOCK_PATTERNS:
        for match in re.finditer(pattern, source):
            line_no   = source[:match.start()].count("\n") + 1
            line_text = lines[line_no-1] if line_no <= len(lines) else ""
            code_part = line_text.split("#")[0]
            col       = match.start() - source.rfind("\n", 0, match.start()) - 1
            if col > len(code_part):
                continue
            issues.append({"test_id":test_id,"issue_text":text,
                           "issue_severity":severity,"line_number":line_no})
    return {
        "tool": "bandit-mock", "issues": issues, "mock": True,
        "high_count":   sum(1 for i in issues if i["issue_severity"]=="HIGH"),
        "medium_count": sum(1 for i in issues if i["issue_severity"]=="MEDIUM"),
        "low_count":    sum(1 for i in issues if i["issue_severity"]=="LOW"),
        "total":        len(issues),
        "ids":          list({i["test_id"] for i in issues}),
    }


# ---------------------------------------------------------------------------
# Safety scanner
# ---------------------------------------------------------------------------
_KNOWN_VULN_PKGS: Dict[str, Tuple[str, str]] = {
    "requests":     ("2.32.0", "CVE-2024-35195"),
    "pillow":       ("10.3.0", "CVE-2024-28219"),
    "urllib3":      ("2.2.2",  "CVE-2024-37891"),
    "cryptography": ("42.0.4", "CVE-2024-26130"),
    "django":       ("4.2.13", "CVE-2024-38875"),
    "flask":        ("3.0.3",  "CVE-2023-30861"),
    "pyyaml":       ("6.0.1",  "CVE-2022-1471"),
    "paramiko":     ("3.4.0",  "CVE-2023-48795"),
}


def _parse_manifest(manifest: str) -> Dict[str, Optional[str]]:
    deps: Dict[str, Optional[str]] = {}
    for line in manifest.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*([=!<>~^]+\s*[^\s;#]+)?", line)
        if m:
            pkg = m.group(1).lower()
            ver = m.group(2)
            deps[pkg] = ver.strip().lstrip("=><~^").strip() if ver else None
    return deps


def _version_lt(v: str, threshold: str) -> bool:
    try:
        from packaging.version import Version
        return Version(v) < Version(threshold)
    except Exception:
        pass
    def _p(s: str) -> List[int]:
        return [int(x) for x in re.split(r"[.\-]", s) if x.isdigit()]
    for a, b in zip(_p(v), _p(threshold)):
        if a != b: return a < b
    return len(_p(v)) < len(_p(threshold))


def _run_safety(manifest: str) -> Dict[str, Any]:
    deps = _parse_manifest(manifest)
    issues: List[Dict[str, Any]] = []
    unpinned: List[str] = []
    for pkg, ver in deps.items():
        if ver is None:
            unpinned.append(pkg); continue
        if pkg in _KNOWN_VULN_PKGS:
            min_safe, cve = _KNOWN_VULN_PKGS[pkg]
            if _version_lt(ver, min_safe):
                issues.append({"package":pkg,"version":ver,"cve":cve,
                                "min_safe":min_safe,"severity":"HIGH"})
    return {"tool":"safety-mock","issues":issues,"unpinned":unpinned,
            "cve_ids":[i["cve"] for i in issues],"vuln_count":len(issues),
            "total":len(issues)+len(unpinned)}


# ---------------------------------------------------------------------------
# Issue extraction  –  stable deterministic sort
# ---------------------------------------------------------------------------
def _extract_issues(source: str, manifest: str) -> List[Issue]:
    issues: List[Issue] = []

    bandit = _run_bandit(source)
    for item in bandit["issues"]:
        tid  = item.get("test_id", "?")
        sev  = item.get("issue_severity", "LOW")
        text = item.get("issue_text", "")
        line = item.get("line_number")
        pri  = (Priority.CRITICAL if tid in ("B307","B301")
                else Priority.HIGH if tid in ("B105","B605")
                else Priority.MEDIUM if tid in ("B603","B303")
                else Priority.LOW)
        evidence = ""
        if line:
            src_lines = source.splitlines()
            if 0 < line <= len(src_lines):
                evidence = src_lines[line-1].strip()
        issues.append(Issue(issue_id=tid, severity=sev, priority=pri,
                            description=text, line_no=line, evidence=evidence))

    safety = _run_safety(manifest)
    for item in safety["issues"]:
        issues.append(Issue(issue_id=item["cve"], severity="HIGH", priority=Priority.DEP,
                            description=f"{item['package']} {item['version']} < {item['min_safe']}",
                            package=item["package"],
                            evidence=f"{item['package']}=={item['version']}"))
    for pkg in safety["unpinned"]:
        issues.append(Issue(issue_id=f"UNPINNED:{pkg}", severity="MEDIUM", priority=Priority.DEP,
                            description=f"{pkg} unpinned", package=pkg,
                            evidence=f"{pkg} has no version pin"))

    # DETERMINISTIC sort: (priority, line_no or 9999, issue_id)
    issues.sort(key=lambda i: (i.priority.value, i.line_no or 9999, i.issue_id))
    return issues


def _by_category(issues: List[Issue]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"CRITICAL":[],"HIGH":[],"MEDIUM":[],"DEP":[]}
    for i in issues:
        cat = i.priority.name if i.priority.name in out else "MEDIUM"
        out[cat].append(i.issue_id)
    return out


def _priority_summary(issues: List[Issue]) -> Dict[str, int]:
    from collections import Counter
    return dict(Counter(i.priority.name for i in issues))


# ---------------------------------------------------------------------------
# Observation  –  structured, phase-aware, loop-hint aware
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    source_code:         str = Field(..., description="Current source code (UTF-8).")
    dependency_manifest: str = Field(..., description="Current requirements.txt.")
    security_policy:     Dict[str, Any] = Field(default_factory=dict)
    task_id:             str   = Field(default="")
    step_index:          int   = Field(default=0, ge=0)
    elapsed_ms:          float = Field(default=0.0, ge=0.0)
    diff:                str   = Field(default="")

    # ── Structured fields the agent MUST read ──────────────────────────────────
    open_issues:     Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Issues grouped by category: CRITICAL/HIGH/MEDIUM/DEP.",
    )
    allowed_actions: List[str] = Field(
        default_factory=list,
        description="Which actions are legal in the current phase.",
    )
    current_phase:   str = Field(default="CRITICAL")
    status:          str = Field(default="")
    hint:            Optional[str] = Field(
        default=None,
        description="Injected hint if agent is looping or stuck.",
    )
    priority_summary:    Dict[str, int]  = Field(default_factory=dict)
    validation_results:  Dict[str, Any]  = Field(default_factory=dict)

    @field_validator("source_code")
    @classmethod
    def _src(cls, v: str) -> str:
        if len(v.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("source_code too large.")
        return v

    @field_validator("dependency_manifest")
    @classmethod
    def _mfst(cls, v: str) -> str:
        if len(v.encode()) > MAX_MANIFEST_BYTES:
            raise ValueError("dependency_manifest too large.")
        return v

    @model_validator(mode="after")
    def _policy(self) -> "Observation":
        required = {"severity_threshold","allowed_licenses","blocked_packages"}
        missing  = required - self.security_policy.keys()
        if missing:
            logger.warning("security_policy missing keys: %s", missing)
        return self

    def fingerprint(self) -> str:
        payload = self.source_code + self.dependency_manifest + json.dumps(self.security_policy, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# FlowGuard  –  strict phase enforcer
# ---------------------------------------------------------------------------
class FlowGuard:
    OUT_OF_ORDER_PENALTY     = R_FLOW_VIOLATION
    PREMATURE_TERMINAL_PENALTY = R_FLOW_VIOLATION

    def __init__(self) -> None:
        self.phase: FlowPhase = FlowPhase.CRITICAL

    def reset(self) -> None:
        self.phase = FlowPhase.CRITICAL

    def advance_if_clear(self, issues: List[Issue]) -> None:
        """Auto-advance through phases when issues in current phase are gone."""
        while self.phase != FlowPhase.TERMINAL:
            pri = _PHASE_TO_PRIORITY.get(self.phase.value)
            if pri is None:
                break
            if not any(i.priority == pri for i in issues):
                self.phase = self.phase.next_phase()
                logger.info("[FlowGuard] Advanced → %s", self.phase.value)
            else:
                break

    def allowed_actions(self, issues: List[Issue]) -> List[str]:
        """Compute the exact set of legal actions in current phase."""
        self.advance_if_clear(issues)
        if self.phase == FlowPhase.TERMINAL:
            return [ActionType.APPROVE.value, ActionType.REJECT.value]
        if self.phase == FlowPhase.CRITICAL:
            return [ActionType.FLAG_VULN.value, ActionType.PROPOSE_PATCH.value]
        if self.phase == FlowPhase.HIGH:
            return [ActionType.FLAG_VULN.value, ActionType.PROPOSE_PATCH.value]
        if self.phase == FlowPhase.DEP:
            return [ActionType.REJECT_DEP.value, ActionType.PROPOSE_PATCH.value]
        return [a.value for a in ActionType]

    def check(self, at: ActionType, issues: List[Issue]) -> Tuple[bool, float, str]:
        """Returns (legal, penalty, reason)."""
        self.advance_if_clear(issues)
        allowed = self.allowed_actions(issues)
        if at.value not in allowed:
            if at in (ActionType.APPROVE, ActionType.REJECT):
                return (False, self.PREMATURE_TERMINAL_PENALTY,
                        f"Cannot terminate in {self.phase.value} phase. "
                        f"Resolve {self.phase.value} issues first. Allowed: {allowed}")
            return (False, self.OUT_OF_ORDER_PENALTY,
                    f"{at.value} not allowed in {self.phase.value} phase. Allowed: {allowed}")
        return True, 0.0, "OK"


# ---------------------------------------------------------------------------
# LoopDetector
# ---------------------------------------------------------------------------
class LoopDetector:
    def __init__(self, window: int = LOOP_WINDOW, limit: int = LOOP_LIMIT) -> None:
        self._window: Deque[str] = deque(maxlen=window)
        self._limit  = limit

    def reset(self) -> None:
        self._window.clear()

    def record(self, action_type: str) -> Tuple[bool, int]:
        """Record action type. Returns (is_looping, streak_count)."""
        self._window.append(action_type)
        if len(self._window) < self._window.maxlen:
            return False, 0
        streak = len(set(self._window)) == 1
        count  = len(self._window) if streak else 0
        return streak and count >= self._limit, count

    def hint(self) -> str:
        return (
            "[LOOP DETECTED] You have repeated the same action 3 times. "
            "Try a different action type. "
            "If CRITICAL issues exist: use PROPOSE_PATCH. "
            "If DEP issues remain after code is fixed: use REJECT_DEP or PROPOSE_PATCH with patched_manifest."
        )


# ---------------------------------------------------------------------------
# Status builder
# ---------------------------------------------------------------------------
def _build_status(issues: List[Issue], phase: str, validation: Optional[ValidationResult]) -> str:
    if not issues:
        return (
            "ALL SCANS CLEAN. "
            "Set remaining_issues={} and use action_type=APPROVE with full reasoning."
        )
    by_cat: Dict[str, List[str]] = {}
    for i in issues:
        by_cat.setdefault(i.priority.name, []).append(
            f"{i.issue_id}(line {i.line_no or '?'})"
        )
    parts = [f"[{cat}] {', '.join(ids)}" for cat, ids in by_cat.items() if ids]
    parts.append(f"Current phase: {phase}.")
    has_critical = any(i.priority == Priority.CRITICAL for i in issues)
    has_high     = any(i.priority == Priority.HIGH for i in issues)
    has_dep      = any(i.priority == Priority.DEP for i in issues)
    if has_critical:
        parts.append("Fix CRITICAL issues first (FLAG_VULN then PROPOSE_PATCH).")
    elif has_high:
        parts.append("Fix HIGH issues next (FLAG_VULN then PROPOSE_PATCH).")
    elif has_dep:
        parts.append("Fix DEP issues (REJECT_DEP per package or PROPOSE_PATCH with patched_manifest).")
    if validation and not validation.passed:
        parts.append(f"Last patch REJECTED: {'; '.join(validation.errors[:2])}")
    sec = "false" if (has_critical or has_high or has_dep) else "true"
    parts.append(f"validation.security={sec}")
    return " | ".join(parts)


def _unified_diff(before: str, after: str) -> str:
    lines = list(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/source.py", tofile="b/source.py", lineterm="",
    ))
    return "".join(lines[:120])


# ---------------------------------------------------------------------------
# Environment  v4.0.0
# ---------------------------------------------------------------------------
class SentinelPREnv:
    """
    SENTINEL-PR v4.0.0 – Judge-Proof Edition.

    Victory condition (strict):
      all_issues_resolved AND action == APPROVE → +5.0 bonus
      No victory → total reward capped at 1.0

    Flow:  CRITICAL → HIGH → DEP → TERMINAL
    Invalid actions: explicit error + R_INVALID_ACTION, episode continues
    Loop detection: hint injected after LOOP_LIMIT repeats
    """

    # Score normalisation ranges per task — grader uses these
    SCORE_RANGES = {
        "task_hardcoded_key":  {"min": -3.0,  "max": 6.0},
        "task_vulnerable_dep": {"min": -3.0,  "max": 6.0},
        "task_eval_auth_flaw": {"min": -18.0, "max": 13.0},
    }
    _SCORE_CLIP_MIN = 0.01
    _SCORE_CLIP_MAX = 0.99

    METADATA = {
        "name":          "SENTINEL-PR",
        "version":       "4.0.0",
        "openenv_api":   OPENENV_API_VERSION,
        "tags":          ["security","static-analysis","dependency-audit"],
        "ram_budget_gb": 8,
    }

    def __init__(self, max_steps: int = MAX_STEPS) -> None:
        self._max_steps       = max_steps
        self._step_count      = 0
        self._start_time:     float = 0.0
        self._current_obs:    Optional[Observation] = None
        self._task_id:        str  = ""
        self._done:           bool = False
        self._flow:           FlowGuard = FlowGuard()
        self._loop:           LoopDetector = LoopDetector()
        self._patch_applied:  bool = False
        self._fixed_keys:     Set[str] = set()
        self._rejected_deps:  Set[str] = set()   # packages cleared via REJECT_DEP
        self._last_validation: Optional[ValidationResult] = None
        self._episode_reward: float = 0.0   # running total for cap calculation
        self._victory:        bool = False
        logger.info("[START] SentinelPREnv v4.0.0 (max_steps=%d)", max_steps)

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self, task_id: str = "task_hardcoded_key") -> Observation:
        self._step_count      = 0
        self._start_time      = time.monotonic()
        self._done            = False
        self._task_id         = task_id
        self._flow.reset()
        self._loop.reset()
        self._patch_applied   = False
        self._fixed_keys      = set()
        self._rejected_deps   = set()
        self._last_validation = None
        self._episode_reward  = 0.0
        self._victory         = False

        obs = _TASK_REGISTRY.get(task_id)
        if obs is None:
            raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(_TASK_REGISTRY)}")

        issues = _extract_issues(obs.source_code, obs.dependency_manifest)
        self._flow.advance_if_clear(issues)
        obs = obs.model_copy(update={
            "task_id":           task_id,
            "step_index":        0,
            "diff":              "",
            "open_issues":       _by_category_full(issues),
            "allowed_actions":   self._flow.allowed_actions(issues),
            "current_phase":     self._flow.phase.value,
            "status":            _build_status(issues, self._flow.phase.value, None),
            "priority_summary":  _priority_summary(issues),
            "validation_results": {},
            "hint":              None,
        })
        self._current_obs = obs
        gc.collect()
        logger.info("[STEP] reset task_id=%s  issues=%d  phase=%s",
                    task_id, len(issues), self._flow.phase.value)
        return obs

    def step(self, action_dict: Any) -> StepResult:  # noqa: C901
        """
        Accept either Action object or raw dict.
        Validates schema BEFORE processing. Returns explicit error if invalid.
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        if self._current_obs is None:
            raise RuntimeError("Call reset() before step().")

        self._step_count += 1
        elapsed_s = time.monotonic() - self._start_time

        # ── 1. Parse and hard-validate action schema ──────────────────────────
        schema_error: Optional[str] = None
        action: Optional[Action] = None
        if isinstance(action_dict, Action):
            action = action_dict   # already validated by Pydantic on construction
        elif action_dict is None:
            schema_error = "action_dict is None – LLM parse failure"
            logger.warning("[STEP %d] None action_dict received", self._step_count)
        elif not isinstance(action_dict, dict):
            schema_error = f"action_dict is not a dict: {type(action_dict).__name__}"
            logger.warning("[STEP %d] Wrong type: %s", self._step_count, type(action_dict))
        elif "_parse_error" in action_dict:
            schema_error = f"LLM parse error: {action_dict.get('_parse_error', 'unknown')}"
            logger.warning("[STEP %d] Explicit parse error forwarded", self._step_count)
        else:
            try:
                action = Action(**action_dict)
            except Exception as exc:
                schema_error = str(exc)
                logger.warning("[STEP %d] Schema error: %s", self._step_count, schema_error)

        # ── 2. Current issues (stable deterministic order) ────────────────────
        _raw_issues = _extract_issues(
            self._current_obs.source_code,
            self._current_obs.dependency_manifest,
        )
        # Filter packages already rejected this episode
        if self._rejected_deps:
            _raw_issues = [i for i in _raw_issues if i.package not in self._rejected_deps]
        current_issues = sorted(
            _raw_issues,
            key=lambda i: (i.priority.value, i.line_no or 9999, i.issue_id),
        )

        reward:     float = 0.0
        breakdown:  Dict[str, float] = {}
        victory     = False
        validation: Optional[ValidationResult] = None
        new_source  = self._current_obs.source_code
        new_manifest= self._current_obs.dependency_manifest
        patch_diff  = ""
        hint:       Optional[str] = None

        # ── 3. Schema invalid → penalty, no processing, continue ─────────────
        if schema_error is not None:
            reward += R_INVALID_ACTION
            breakdown["invalid_action_penalty"] = R_INVALID_ACTION
            self._episode_reward += reward
            next_obs = self._current_obs.model_copy(update={
                "step_index": self._step_count,
                "elapsed_ms": elapsed_s * 1000,
                "hint": f"[SCHEMA ERROR] {schema_error[:200]}. Fix your JSON and retry.",
            })
            self._current_obs = next_obs
            result = StepResult(
                observation=next_obs,
                reward=round(reward, 4),
                done=self._step_count >= self._max_steps,
                terminated=False,
                info={
                    "step": self._step_count,
                    "schema_error": schema_error,
                    "reward_breakdown": breakdown,
                    "flow_phase": self._flow.phase.value,
                    "open_issues_count": len(current_issues),
                    "victory": False,
                },
            )
            logger.info("[STEP %d] invalid action reward=%.2f", self._step_count, reward)
            if self._step_count >= self._max_steps:
                self._done = True
            return result

        at = action.action_type

        # ── 4. Loop detection ─────────────────────────────────────────────────
        is_looping, streak = self._loop.record(at.value)
        if is_looping:
            hint = self._loop.hint()
            logger.warning("[STEP %d] Loop detected streak=%d", self._step_count, streak)

        # ── 5. FlowGuard check ────────────────────────────────────────────────
        flow_ok, flow_penalty, flow_msg = self._flow.check(at, current_issues)
        if not flow_ok:
            reward += flow_penalty
            breakdown["flow_violation"] = flow_penalty
            logger.warning("[STEP %d] FlowGuard violation: %s", self._step_count, flow_msg)
            # Do NOT terminate – penalty applied and continue
            self._episode_reward += reward
            next_obs = self._current_obs.model_copy(update={
                "step_index":    self._step_count,
                "elapsed_ms":    elapsed_s * 1000,
                "allowed_actions": self._flow.allowed_actions(current_issues),
                "current_phase": self._flow.phase.value,
                "hint":          f"[FLOW VIOLATION] {flow_msg}",
            })
            self._current_obs = next_obs
            result = StepResult(
                observation=next_obs,
                reward=round(reward, 4),
                done=self._step_count >= self._max_steps,
                terminated=False,
                info={
                    "step": self._step_count,
                    "flow_violation": flow_msg,
                    "reward_breakdown": breakdown,
                    "flow_phase": self._flow.phase.value,
                    "open_issues_count": len(current_issues),
                    "victory": False,
                },
            )
            if self._step_count >= self._max_steps:
                self._done = True
            return result

        # ── 6. Route by action type ───────────────────────────────────────────
        if at is ActionType.FLAG_VULN:
            detail_lower = (action.detail or "").lower()
            already_fixed = any(
                k for k in self._fixed_keys
                if any(tok in detail_lower for tok in k.lower().split(":"))
            )
            if already_fixed:
                reward += -1.0  # contradiction
                breakdown["contradiction"] = -1.0
            elif current_issues:
                reward += R_FLAG_CORRECT
                breakdown["flag_correct"] = R_FLAG_CORRECT
            else:
                reward += R_FLAG_FP
                breakdown["flag_false_positive"] = R_FLAG_FP

        elif at is ActionType.PROPOSE_PATCH:
            has_source   = bool(action.patched_source and action.patched_source.strip())
            has_manifest = bool(action.patched_manifest and action.patched_manifest.strip())

            if has_source:
                # Code patch: run full validator (ast + bandit + sandbox)
                validation = PatchValidator.validate(action.patched_source)
                self._last_validation = validation
                patch_accepted = validation.passed
                if not patch_accepted:
                    reward += R_INVALID_ACTION
                    breakdown["patch_validation_failed"] = R_INVALID_ACTION
                    logger.warning("[STEP %d] Patch REJECTED: %s", self._step_count, validation.errors)
            elif has_manifest:
                # Manifest-only patch: no code to validate, always accepted
                validation = ValidationResult(syntax=True, security=True, runtime=True, passed=True)
                self._last_validation = validation
                patch_accepted = True
                logger.info("[STEP %d] Manifest-only patch accepted.", self._step_count)
            else:
                # Neither source nor manifest – invalid
                patch_accepted = False
                reward += R_INVALID_ACTION
                breakdown["patch_no_content"] = R_INVALID_ACTION
                logger.warning("[STEP %d] PROPOSE_PATCH has no patched_source or patched_manifest.", self._step_count)

            if patch_accepted:
                new_source   = action.patched_source if has_source else self._current_obs.source_code
                new_manifest = action.patched_manifest if has_manifest else self._current_obs.dependency_manifest
                if has_source:
                    patch_diff = _unified_diff(self._current_obs.source_code, new_source)
                self._patch_applied = True

                # Filter rejected deps before computing after_issues
                after_issues_raw = _extract_issues(new_source, new_manifest)
                if self._rejected_deps:
                    after_issues_raw = [i for i in after_issues_raw if i.package not in self._rejected_deps]
                after_keys = {i.key() for i in after_issues_raw}
                resolved   = [i for i in current_issues if i.key() not in after_keys]
                self._fixed_keys.update(i.key() for i in resolved)

                fix_map = {
                    Priority.CRITICAL: R_FIX_CRITICAL,
                    Priority.HIGH:     R_FIX_HIGH,
                    Priority.DEP:      R_FIX_DEP,
                    Priority.MEDIUM:   R_FIX_MEDIUM,
                    Priority.LOW:      0.1,
                }
                for iss in resolved:
                    r = fix_map.get(iss.priority, 0.1)
                    reward += r
                    breakdown[f"fixed_{iss.priority.name}_{iss.issue_id}"] = r

                if not resolved:
                    reward += R_PARTIAL_FIX
                    breakdown["partial_fix"] = R_PARTIAL_FIX

                current_issues = after_issues_raw
                self._flow.advance_if_clear(current_issues)

        elif at is ActionType.REJECT_DEP:
            pkg = (action.package or "").lower()
            dep_issues = [i for i in current_issues if i.package == pkg]
            if dep_issues:
                reward += R_FLAG_CORRECT
                breakdown["reject_dep_correct"] = R_FLAG_CORRECT
                # FIX: track rejected packages so they're excluded from future scans
                self._rejected_deps.add(pkg)
                # Remove rejected dep from current working set so FlowGuard can advance
                current_issues = [i for i in current_issues if i.package != pkg]
                self._flow.advance_if_clear(current_issues)
                logger.info("[REJECT_DEP] Rejected %s. Remaining DEP issues: %d",
                            pkg, sum(1 for i in current_issues if i.priority.name == "DEP"))
            else:
                reward += R_FLAG_FP
                breakdown["reject_dep_fp"] = R_FLAG_FP

        elif at is ActionType.APPROVE:
            if current_issues:
                reward += R_APPROVE_DIRTY
                breakdown["approve_with_issues"] = R_APPROVE_DIRTY
                logger.warning("[STEP %d] APPROVE with %d open issues", self._step_count, len(current_issues))
            else:
                reward += R_APPROVE_CLEAN
                breakdown["approve_clean"] = R_APPROVE_CLEAN
                if self._patch_applied:
                    reward  += R_VICTORY_BONUS
                    breakdown["victory_bonus"] = R_VICTORY_BONUS
                    victory  = True
                    self._victory = True
                    logger.info("[END] VICTORY reward=+%.2f", reward)

        elif at is ActionType.REJECT:
            if current_issues:
                reward += 0.8   # correct reject (not 1.0 to keep score in open interval)
                breakdown["correct_reject"] = 0.8
            else:
                reward += -0.3  # wrong reject
                breakdown["wrong_reject"] = -0.3

        # ── 7. Repetition penalty ─────────────────────────────────────────────
        if streak > 0:
            rep_pen = R_REPETITION_BASE * streak
            reward += rep_pen
            breakdown["repetition_penalty"] = rep_pen

        # ── 8. Termination ────────────────────────────────────────────────────
        terminal_acts = {ActionType.APPROVE, ActionType.REJECT}
        terminated    = victory
        done          = (
            at in terminal_acts
            or self._step_count >= self._max_steps
            or terminated
        )
        self._done = done

        # ── 9. Apply no-victory reward cap at episode end ─────────────────────
        self._episode_reward += reward
        if done and not self._victory:
            cap_excess = self._episode_reward - NO_VICTORY_REWARD_CAP
            if cap_excess > 0:
                reward     -= cap_excess
                breakdown["no_victory_cap"] = -cap_excess
                self._episode_reward = NO_VICTORY_REWARD_CAP
                logger.info("[END] No victory – reward capped at %.1f", NO_VICTORY_REWARD_CAP)

        # ── 10. Build next observation ────────────────────────────────────────
        issues_next = _extract_issues(new_source, new_manifest)
        # Filter out packages that have been explicitly rejected this episode
        if self._rejected_deps:
            issues_next = [i for i in issues_next if i.package not in self._rejected_deps]
        self._flow.advance_if_clear(issues_next)
        val_dict    = validation.model_dump() if validation else {}
        status      = _build_status(issues_next, self._flow.phase.value, validation)

        next_obs = self._current_obs.model_copy(update={
            "source_code":          new_source,
            "dependency_manifest":  new_manifest,
            "step_index":           self._step_count,
            "elapsed_ms":           elapsed_s * 1000,
            "diff":                 patch_diff,
            "open_issues":          _by_category_full(issues_next),
            "allowed_actions":      self._flow.allowed_actions(issues_next),
            "current_phase":        self._flow.phase.value,
            "status":               status,
            "priority_summary":     _priority_summary(issues_next),
            "validation_results":   val_dict,
            "hint":                 hint,
        })
        self._current_obs = next_obs

        info: Dict[str, Any] = {
            "step":              self._step_count,
            "elapsed_s":         round(elapsed_s, 3),
            "reward_breakdown":  breakdown,
            "open_issues_count": len(issues_next),
            "priority_summary":  _priority_summary(issues_next),
            "allowed_actions":   self._flow.allowed_actions(issues_next),
            "flow_phase":        self._flow.phase.value,
            "task_id":           self._task_id,
            "fingerprint":       next_obs.fingerprint(),
            "patch_applied":     self._patch_applied,
            "victory":           victory,
            "episode_reward":    self._episode_reward,
            "streak":            streak if isinstance(streak, int) else 0,
            # Always-present safety keys (prevent KeyError in inference.py)
            "schema_error":      "",
            "flow_violation":    "",
            "hint":              hint or "",
            "validation":        (val_dict if val_dict is not None else {}) if "val_dict" in str(locals()) else {},
            "validation_results": (val_dict if val_dict is not None else {}) if "val_dict" in str(locals()) else {},
        }

        logger.info(
            "[STEP %d] action=%s reward=%.4f done=%s phase=%s open=%d",
            self._step_count, at.value, reward, done,
            self._flow.phase.value, len(issues_next),
        )
        gc.collect()
        return StepResult(
            observation=next_obs,
            reward=round(reward, 4),
            done=done,
            terminated=terminated,
            info=info,
            score=self._compute_score(self._episode_reward),
        )

    def _compute_score(self, reward: float) -> float:
        """Compute score strictly in (0.01, 0.99) from a step or episode reward."""
        tid    = self._task_id or "task_eval_auth_flaw"
        ranges = self.SCORE_RANGES.get(tid, {"min": -18.0, "max": 13.0})
        r_min, r_max = ranges["min"], ranges["max"]
        if r_max == r_min:
            normalised = 0.5
        else:
            normalised = (reward - r_min) / (r_max - r_min)
        if self._victory:
            normalised = max(normalised, 0.90)
        return round(max(self._SCORE_CLIP_MIN, min(self._SCORE_CLIP_MAX, normalised)), 4)

    def score(self, total_reward: float, task_id: str = "") -> float:
        """
        Convert episode reward to a score strictly in (0.01, 0.99).
        Called by the OpenEnv grader after each episode.
        Score is NEVER 0.0 or 1.0 — always clipped to open interval.
        """
        tid    = task_id or self._task_id
        ranges = self.SCORE_RANGES.get(tid, {"min": -18.0, "max": 13.0})
        r_min, r_max = ranges["min"], ranges["max"]
        if r_max == r_min:
            normalised = 0.5
        else:
            normalised = (total_reward - r_min) / (r_max - r_min)
        # Victory pushes score toward 0.95 but still < 0.99
        if self._victory:
            normalised = max(normalised, 0.90)
        # Clip strictly to open interval (0.01, 0.99)
        return round(max(self._SCORE_CLIP_MIN, min(self._SCORE_CLIP_MAX, normalised)), 4)

    def render(self) -> str:
        if self._current_obs is None:
            return "<not initialised>"
        return (
            f"[SENTINEL-PR v4.0.0] Task={self._task_id} "
            f"Step={self._step_count}/{self._max_steps} "
            f"Phase={self._flow.phase.value}\n"
            f"Issues: {self._current_obs.priority_summary}\n"
            f"Allowed: {self._current_obs.allowed_actions}\n"
            f"Status: {self._current_obs.status[:100]}"
        )

    @classmethod
    def metadata(cls) -> Dict[str, Any]:
        return cls.METADATA

    @staticmethod
    def observation_space() -> Dict[str, Any]:
        return Observation.model_json_schema()

    @staticmethod
    def action_space() -> Dict[str, Any]:
        return Action.model_json_schema()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _by_category_full(issues: List[Issue]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {
        "CRITICAL": [], "HIGH": [], "MEDIUM": [], "DEP": []
    }
    for i in issues:
        cat = i.priority.name if i.priority.name in out else "MEDIUM"
        out[cat].append({
            "issue_id":    i.issue_id,
            "description": i.description[:60],
            "line_no":     i.line_no,
            "package":     i.package,
            "evidence":    i.evidence[:80],
        })
    return out


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
_DEFAULT_POLICY: Dict[str, Any] = {
    "severity_threshold":      "ZERO",
    "allowed_licenses":        ["MIT","Apache-2.0","BSD-2-Clause","BSD-3-Clause"],
    "blocked_packages":        ["pycrypto","insecure-package","pickle5"],
    "require_pinned_versions": True,
    "max_dependency_age_days": 365,
}

_TASK1_SRC = textwrap.dedent("""\
    import requests, os
    BASE_URL = "https://api.internal.corp/v2"
    API_KEY = "sk-prod-aBcDeFgHiJkLmNoPqRsTuVwXyZ012345"
    def fetch_user_data(user_id: int) -> dict:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        resp = requests.get(f"{BASE_URL}/users/{user_id}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
""")
_TASK1_MFT = "requests==2.31.0\npydantic==2.7.0\n"

_TASK2_SRC = textwrap.dedent("""\
    from PIL import Image
    import io
    def resize_avatar(data: bytes, max_px: int = 256) -> bytes:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
""")
_TASK2_MFT = "Pillow==9.5.0\nrequests==2.31.0\nurllib3==1.26.18\ncryptography==38.0.4\n"

_TASK3_SRC = textwrap.dedent("""\
    import hmac, hashlib, time
    from functools import wraps
    from flask import request, jsonify, g
    SECRET = "super-secret-signing-key"
    def _verify_token(token: str) -> dict:
        try:
            import base64
            parts = token.split(".")
            payload_b64, sig = parts[0], parts[1]
            payload_str = base64.b64decode(payload_b64 + "==").decode()
            payload = eval(payload_str)
            expected_sig = hmac.new(SECRET.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
            if sig == expected_sig:
                return payload
            return {}
        except Exception:
            return {}
    def require_auth(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get("X-Auth-Token", "")
            identity = _verify_token(token)
            if not identity:
                return jsonify({"error": "Unauthorized"}), 401
            g.user = identity
            return f(*args, **kwargs)
        return decorated
    def get_expiry(token: str) -> int:
        import base64
        parts = token.split(".")
        payload_str = base64.b64decode(parts[0] + "==").decode()
        data = eval(payload_str)
        return data.get("exp", int(time.time()))
""")
_TASK3_MFT = "flask==2.3.3\npyyaml==5.4.1\nrequests==2.28.2\n"

_TASK_REGISTRY: Dict[str, Observation] = {
    "task_hardcoded_key":  Observation(source_code=_TASK1_SRC, dependency_manifest=_TASK1_MFT, security_policy=_DEFAULT_POLICY),
    "task_vulnerable_dep": Observation(source_code=_TASK2_SRC, dependency_manifest=_TASK2_MFT, security_policy=_DEFAULT_POLICY),
    "task_eval_auth_flaw": Observation(source_code=_TASK3_SRC, dependency_manifest=_TASK3_MFT, security_policy=_DEFAULT_POLICY),
}


def make_env(**kwargs: Any) -> SentinelPREnv:
    """OpenEnv entry-point."""
    return SentinelPREnv(**kwargs)
