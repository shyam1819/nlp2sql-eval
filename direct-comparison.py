"""
Deterministic, grain-level SQL evaluation metrics for DeepEval.

Decomposes predicted SQL (LLMTestCase.actual_output) vs gold SQL
(LLMTestCase.expected_output) into structural "grains" via sqlglot and scores
each with set precision/recall/F1. No LLM => fully deterministic scores.

Grains: tables, columns, joins, group_by, filters, aggregations, arithmetic.

Designed to slot into the layered cascade:
  Layer 0  SQLValidityMetric     (parses / executes?)
  Layer 2  SQLGrainMetric        (aggregate "what went wrong" score)
           make_grain_metrics()  (one metric per grain -> per-grain report rows)
Execution accuracy (Layer 1) stays your correctness arbiter; these are
diagnostics. Pair with a scoped-down GEval for intent (Layer 3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import sqlglot
from sqlglot import exp, parse_one
from sqlglot.optimizer.qualify import qualify

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GRAINS = ("tables", "columns", "joins", "group_by", "filters", "aggregations", "arithmetic")
DEFAULT_WEIGHTS = {
    "tables": 0.25, "columns": 0.20, "joins": 0.15, "group_by": 0.10,
    "filters": 0.15, "aggregations": 0.10, "arithmetic": 0.05,
}
ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _canonicalize_ctes(ast, dialect: str):
    """Rename CTEs to positional tokens __cte0__.. (by definition order) and
    rewrite references, so CTE naming differences don't cause false mismatches."""
    mapping = {c.alias_or_name: f"__cte{i}__"
               for i, c in enumerate(ast.find_all(exp.CTE))}
    if not mapping:
        return ast
    for cte in ast.find_all(exp.CTE):
        if cte.alias_or_name in mapping:
            cte.set("alias", exp.TableAlias(this=exp.to_identifier(mapping[cte.alias_or_name])))
    for t in ast.find_all(exp.Table):
        if t.name in mapping:
            t.set("this", exp.to_identifier(mapping[t.name]))
    return ast, set(mapping.values())


def extract_grains(sql: str, dialect: str = "databricks",
                   schema: Optional[dict] = None) -> Dict[str, Set[str]]:
    ast = parse_one(sql, dialect=dialect)
    if schema is not None:
        try:
            ast = qualify(ast, schema=schema, dialect=dialect)
        except Exception:
            pass
    ast, cte_tokens = _canonicalize_ctes(ast, dialect)
    return {
        "tables": {t.name for t in ast.find_all(exp.Table)} - cte_tokens,
        "columns": {c.name for c in ast.find_all(exp.Column)},
        "joins": {_norm(j.sql(dialect=dialect)) for j in ast.find_all(exp.Join)},
        "group_by": {_norm(g.sql(dialect=dialect)) for g in ast.find_all(exp.Group)},
        "filters": {_norm(w.this.sql(dialect=dialect)) for w in ast.find_all(exp.Where) if w.this},
        "aggregations": {_norm(a.sql(dialect=dialect)) for a in ast.find_all(exp.AggFunc)},
        "arithmetic": {_norm(n.sql(dialect=dialect)) for n in ast.find_all(ARITHMETIC)},
    }


@dataclass
class GrainResult:
    precision: float
    recall: float
    f1: float
    missing: Set[str] = field(default_factory=set)
    extra: Set[str] = field(default_factory=set)


def _prf(pred: Set[str], gold: Set[str]) -> GrainResult:
    if not pred and not gold:
        return GrainResult(1.0, 1.0, 1.0)
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return GrainResult(precision, recall, f1, gold - pred, pred - gold)


def grain_analysis(pred_sql: str, gold_sql: str, dialect: str = "databricks",
                   schema: Optional[dict] = None) -> Dict[str, GrainResult]:
    p = extract_grains(pred_sql, dialect, schema)
    g = extract_grains(gold_sql, dialect, schema)
    return {grain: _prf(p[grain], g[grain]) for grain in GRAINS}


def weighted_score(results, weights=DEFAULT_WEIGHTS) -> float:
    tot = sum(weights.get(k, 0) for k in results)
    return sum(results[k].f1 * weights.get(k, 0) for k in results) / tot if tot else 0.0


# ---------------------------------------------------------------------------
# DeepEval metrics
# ---------------------------------------------------------------------------
class SQLGrainMetric(BaseMetric):
    """Aggregate grain F1 across all grains, with a per-grain diagnostic reason."""

    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]

    def __init__(self, threshold: float = 0.8, dialect: str = "databricks",
                 schema: Optional[dict] = None, weights: dict = DEFAULT_WEIGHTS,
                 include_reason: bool = True):
        self.threshold = threshold
        self.dialect = dialect
        self.schema = schema
        self.weights = weights
        self.include_reason = include_reason
        self.async_mode = False
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            results = grain_analysis(test_case.actual_output, test_case.expected_output,
                                     self.dialect, self.schema)
            self.score = weighted_score(results, self.weights)
            if self.include_reason:
                lines = []
                for grain, r in results.items():
                    tag = "OK " if r.f1 == 1.0 else "!! "
                    detail = ""
                    if r.missing:
                        detail += f" missing={sorted(r.missing)}"
                    if r.extra:
                        detail += f" extra={sorted(r.extra)}"
                    lines.append(f"{tag}{grain}: F1={r.f1:.2f}{detail}")
                self.reason = "Grain breakdown -> " + " | ".join(lines)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = f"SQL parse/analysis failed: {e}"
            self.score = 0.0
            self.success = False
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "SQL Grain (aggregate)"


class _SingleGrainMetric(BaseMetric):
    """One grain, scored as its F1. Lets each grain appear as its own report row."""
    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]

    def __init__(self, grain: str, threshold: float = 0.8,
                 dialect: str = "databricks", schema: Optional[dict] = None):
        self.grain = grain
        self.threshold = threshold
        self.dialect = dialect
        self.schema = schema
        self.include_reason = True
        self.async_mode = False
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            r = grain_analysis(test_case.actual_output, test_case.expected_output,
                               self.dialect, self.schema)[self.grain]
            self.score = r.f1
            bits = [f"P={r.precision:.2f} R={r.recall:.2f}"]
            if r.missing:
                bits.append(f"missing={sorted(r.missing)}")
            if r.extra:
                bits.append(f"extra={sorted(r.extra)}")
            self.reason = "; ".join(bits)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = f"SQL parse/analysis failed: {e}"
            self.score = 0.0
            self.success = False
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return f"SQL Grain: {self.grain}"


def make_grain_metrics(grains=GRAINS, threshold: float = 0.8,
                       dialect: str = "databricks",
                       schema: Optional[dict] = None) -> List[BaseMetric]:
    return [_SingleGrainMetric(g, threshold, dialect, schema) for g in grains]


class SQLValidityMetric(BaseMetric):
    """Layer 0: does the predicted SQL parse in the target dialect? (1.0/0.0)"""
    _required_params = [LLMTestCaseParams.ACTUAL_OUTPUT]

    def __init__(self, dialect: str = "databricks"):
        self.threshold = 1.0
        self.dialect = dialect
        self.include_reason = True
        self.async_mode = False
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            parse_one(test_case.actual_output, dialect=self.dialect)
            self.score, self.reason = 1.0, "Parses cleanly."
        except Exception as e:
            self.score, self.reason = 0.0, f"Parse error: {e}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        self.success = (self.error is None) and (self.score >= self.threshold)
        return self.success

    @property
    def __name__(self):
        return "SQL Validity"
