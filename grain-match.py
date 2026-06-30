"""
GRAIN evaluation in LLM-judge format (DeepEval).

Same grain taxonomy as the deterministic sql_grain_eval.py
(tables, columns, joins, group_by, filters, aggregations, arithmetic) + intent,
but each grain is scored by an LLM judge instead of by sqlglot set-matching.

Why this version: an LLM judges grains *semantically*, so it does NOT punish
stylistically-different-but-equivalent SQL (the false-negative weakness of the
deterministic grains). Trade-off: it's stochastic -> pair with temperature=0
and/or the StableGEval wrapper from sql_llm_eval.py.

Two forms:
  1. SQLGrainJudge      - ONE LLM call returns all grain scores (token-efficient,
                          structured pydantic output, single unified reason).
  2. make_grain_gevals  - one GEval per grain (native multi-criteria report rows),
                          the idiomatic DeepEval way you originally asked about.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from deepeval.metrics import BaseMetric, GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams as P

GRAINS = ("tables", "columns", "joins", "group_by", "filters", "aggregations", "arithmetic")
DEFAULT_WEIGHTS = {
    "tables": 0.25, "columns": 0.20, "joins": 0.15, "group_by": 0.10,
    "filters": 0.15, "aggregations": 0.10, "arithmetic": 0.05,
}

# Per-grain instructions shared by BOTH forms, so the two stay comparable.
GRAIN_GUIDANCE = {
    "tables":       "the SAME base tables are read (ignore CTE/alias NAMES; judge the underlying tables).",
    "columns":      "the SAME columns are selected/projected for the user's need.",
    "joins":        "joins connect the right tables on the right keys with the right join TYPE (inner/left/etc.), regardless of syntax (JOIN vs IN/EXISTS).",
    "group_by":     "the grouping is at the SAME grain (same group-by columns).",
    "filters":      "the WHERE/HAVING conditions are logically equivalent (e.g. `age>25` == `25<age`); penalize missing or extra filters.",
    "aggregations": "the SAME aggregate functions are applied to the SAME columns (SUM/COUNT/AVG etc.).",
    "arithmetic":   "arithmetic/derived expressions are equivalent; penalize spurious or missing math.",
}


# ---------------------------------------------------------------------------
# Form 1: single-call structured grain judge
# ---------------------------------------------------------------------------
class _GrainScore(BaseModel):
    score: int = Field(ge=0, le=10, description="0=wrong, 10=fully correct")
    reason: str

class SQLGrainVerdict(BaseModel):
    tables: _GrainScore
    columns: _GrainScore
    joins: _GrainScore
    group_by: _GrainScore
    filters: _GrainScore
    aggregations: _GrainScore
    arithmetic: _GrainScore
    intent: _GrainScore        # does it answer the user's question?
    query_match: _GrainScore   # overall SQL match vs gold (semantic equivalence)


# Rubrics for the two query-level dimensions (your original two metrics).
INTENT_RUBRIC = [
    Rubric(score_range=(0, 2),  expected_outcome="Answers a different question than asked."),
    Rubric(score_range=(3, 6),  expected_outcome="Addresses the question but misses part of the intent."),
    Rubric(score_range=(7, 9),  expected_outcome="Answers the intent with a minor gap."),
    Rubric(score_range=(10, 10),expected_outcome="Fully answers the user's analytical intent."),
]
QUERY_MATCH_RUBRIC = [
    Rubric(score_range=(0, 2),  expected_outcome="Would return a different result set than gold."),
    Rubric(score_range=(3, 6),  expected_outcome="Partially overlapping results; notable differences."),
    Rubric(score_range=(7, 9),  expected_outcome="Same results except a minor edge case."),
    Rubric(score_range=(10, 10),expected_outcome="Semantically equivalent to gold for any data state."),
]


def _bands_from(rubrics: List[Rubric]) -> str:
    out = []
    for r in rubrics:
        lo, hi = r.score_range
        rng = f"{lo}" if lo == hi else f"{lo}-{hi}"
        out.append(f"{rng}={r.expected_outcome}")
    return "; ".join(out)


def _render_bands(grain: str) -> str:
    return _bands_from(GRAIN_RUBRICS[grain])


def _build_prompt(question: str, gold: str, pred: str, schema: Optional[str]) -> str:
    grain_lines = "\n".join(
        f"- {g}: judge whether {GRAIN_GUIDANCE[g]}\n    bands -> {_render_bands(g)}"
        for g in GRAINS
    )
    schema_block = f"\n[SCHEMA]\n{schema}\n" if schema else ""
    return f"""You are a strict SQL evaluator. Compare a PREDICTED query against a GOLD \
reference. Score the structural grains AND two query-level dimensions, each 0-10 \
using its own scoring bands, with a one-sentence reason. Base grain scores on the \
PROPORTION of that grain's elements that are correct. Judge SEMANTICS, not syntax: \
do not penalize equivalent rewrites, alias/CTE renaming, predicate reordering, or \
column order.

[USER QUESTION]
{question}
{schema_block}
[GOLD SQL]
{gold}

[PREDICTED SQL]
{pred}

Score these grains, each against its own bands:
{grain_lines}

Then score these two query-level dimensions:
- intent: does the predicted query answer the user's actual question?
    bands -> {_bands_from(INTENT_RUBRIC)}
- query_match: overall, would the predicted query match the gold's result?
    bands -> {_bands_from(QUERY_MATCH_RUBRIC)}

Return your verdict for every grain and both query-level dimensions."""


class SQLGrainJudge(BaseMetric):
    """One LLM call -> 7 grain scores + intent + query_match, combined into a
    configurable composite. `judge` is a DeepEvalBaseLLM (e.g. GPTModel(temperature=0)).

    After measure(), exposes:
      .grain_scores (dict), .grain_aggregate, .intent_score,
      .query_match_score, and .score (the composite).

    component_weights blends the three groups into the headline score, e.g.
    {"grains": 0.5, "intent": 0.2, "query_match": 0.3}.
    """

    DEFAULT_COMPONENT_WEIGHTS = {"grains": 0.5, "intent": 0.2, "query_match": 0.3}

    def __init__(self, judge, threshold: float = 0.8,
                 weights: dict = DEFAULT_WEIGHTS, schema: Optional[str] = None,
                 component_weights: Optional[dict] = None):
        self.judge = judge
        self.threshold = threshold
        self.weights = weights                                   # within-grain weights
        self.component_weights = component_weights or self.DEFAULT_COMPONENT_WEIGHTS
        self.schema = schema
        self.async_mode = False
        self.include_reason = True
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False
        self.grain_scores: Dict[str, float] = {}
        self.grain_aggregate: float = 0.0
        self.intent_score: float = 0.0
        self.query_match_score: float = 0.0

    def _consume(self, verdict) -> float:
        # verdict may be a pydantic obj (structured output) or a JSON string
        if isinstance(verdict, str):
            import json, re
            verdict = SQLGrainVerdict(**json.loads(re.sub(r"```(json)?", "", verdict).strip()))
        self.grain_scores = {g: getattr(verdict, g).score / 10.0 for g in GRAINS}
        self.intent_score = verdict.intent.score / 10.0
        self.query_match_score = verdict.query_match.score / 10.0

        gtot = sum(self.weights.get(g, 0) for g in GRAINS)
        self.grain_aggregate = (
            sum(self.grain_scores[g] * self.weights.get(g, 0) for g in GRAINS) / gtot
            if gtot else 0.0
        )

        cw = self.component_weights
        parts_w = {"grains": self.grain_aggregate,
                   "intent": self.intent_score,
                   "query_match": self.query_match_score}
        ctot = sum(cw.get(k, 0) for k in parts_w)
        self.score = (sum(parts_w[k] * cw.get(k, 0) for k in parts_w) / ctot
                      if ctot else 0.0)

        if self.include_reason:
            parts = []
            for g in GRAINS:
                tag = "OK " if self.grain_scores[g] >= 0.9 else "!! "
                parts.append(f"{tag}{g}={self.grain_scores[g]:.1f} ({getattr(verdict, g).reason})")
            parts.append(f"[grain_agg={self.grain_aggregate:.2f}]")
            parts.append(f"intent={self.intent_score:.1f} ({verdict.intent.reason})")
            parts.append(f"query_match={self.query_match_score:.1f} ({verdict.query_match.reason})")
            self.reason = " | ".join(parts)
        self.success = self.score >= self.threshold
        return self.score

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            prompt = _build_prompt(test_case.input, test_case.expected_output,
                                   test_case.actual_output, self.schema)
            verdict, _cost = self.judge.generate(prompt, schema=SQLGrainVerdict)
            return self._consume(verdict)
        except Exception as e:
            self.error = str(e); self.success = False; raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            prompt = _build_prompt(test_case.input, test_case.expected_output,
                                   test_case.actual_output, self.schema)
            verdict, _cost = await self.judge.a_generate(prompt, schema=SQLGrainVerdict)
            return self._consume(verdict)
        except Exception as e:
            self.error = str(e); self.success = False; raise

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
        return "SQL Grain Judge (LLM, all grains)"


# ---------------------------------------------------------------------------
# Form 2: one GEval per grain (native multi-criteria report rows)
# ---------------------------------------------------------------------------
# Per-grain rubrics: banded by the PROPORTION of that grain's elements that are
# correct (e.g. "3 of 4 tables"). Override any of these to tune banding.
GRAIN_RUBRICS: Dict[str, List[Rubric]] = {
    "tables": [
        Rubric(score_range=(0, 2),  expected_outcome="Most required tables missing or wrong table(s) used."),
        Rubric(score_range=(3, 6),  expected_outcome="About half the required tables correct (e.g. 2 of 4)."),
        Rubric(score_range=(7, 9),  expected_outcome="All but one table correct (e.g. 3 of 4), or one spurious table."),
        Rubric(score_range=(10, 10),expected_outcome="All required base tables present, no extra tables."),
    ],
    "columns": [
        Rubric(score_range=(0, 2),  expected_outcome="Most needed columns missing or wrong columns selected."),
        Rubric(score_range=(3, 6),  expected_outcome="Roughly half the required columns correct."),
        Rubric(score_range=(7, 9),  expected_outcome="Nearly all columns correct; one missing or one extra."),
        Rubric(score_range=(10, 10),expected_outcome="Exactly the required columns, none missing or extra."),
    ],
    "joins": [
        Rubric(score_range=(0, 2),  expected_outcome="Wrong tables joined or a required join missing."),
        Rubric(score_range=(3, 6),  expected_outcome="Right tables but wrong key(s) or wrong join type."),
        Rubric(score_range=(7, 9),  expected_outcome="All joins correct but one minor condition/type imperfection."),
        Rubric(score_range=(10, 10),expected_outcome="All joins on correct keys with correct join types."),
    ],
    "group_by": [
        Rubric(score_range=(0, 2),  expected_outcome="Grouping grain wrong or missing where required."),
        Rubric(score_range=(3, 6),  expected_outcome="Partially correct grouping (one key off)."),
        Rubric(score_range=(7, 9),  expected_outcome="Correct grain with a minor extra/missing key."),
        Rubric(score_range=(10, 10),expected_outcome="Exact grouping grain matches gold."),
    ],
    "filters": [
        Rubric(score_range=(0, 2),  expected_outcome="Required filters missing or logically wrong."),
        Rubric(score_range=(3, 6),  expected_outcome="About half the filter conditions correct."),
        Rubric(score_range=(7, 9),  expected_outcome="All but one predicate correct (e.g. 2 of 3 conditions)."),
        Rubric(score_range=(10, 10),expected_outcome="All filter predicates logically equivalent to gold."),
    ],
    "aggregations": [
        Rubric(score_range=(0, 2),  expected_outcome="Wrong aggregate functions or applied to wrong columns."),
        Rubric(score_range=(3, 6),  expected_outcome="About half the aggregations correct."),
        Rubric(score_range=(7, 9),  expected_outcome="Nearly all aggregations correct; one off."),
        Rubric(score_range=(10, 10),expected_outcome="All aggregate functions on the correct columns."),
    ],
    "arithmetic": [
        Rubric(score_range=(0, 2),  expected_outcome="Required derived expression missing or spurious math added."),
        Rubric(score_range=(3, 6),  expected_outcome="Partially correct arithmetic."),
        Rubric(score_range=(7, 9),  expected_outcome="Correct except one minor term."),
        Rubric(score_range=(10, 10),expected_outcome="All arithmetic/derived expressions equivalent (or none needed)."),
    ],
}

def grain_geval(grain: str, threshold: float = 0.8, judge=None,
                rubric: Optional[List[Rubric]] = None) -> GEval:
    return GEval(
        name=f"Grain: {grain}",
        evaluation_steps=[
            f"Treat 'actual output' as predicted SQL and 'expected output' as gold SQL.",
            f"Focusing ONLY on the {grain} grain, judge whether {GRAIN_GUIDANCE[grain]}",
            "Judge semantics, not syntax; ignore equivalent rewrites and alias/CTE names.",
            "Base the score on the PROPORTION of this grain's elements that are correct, per the rubric.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=rubric or GRAIN_RUBRICS[grain],
        threshold=threshold,
        model=judge,
    )

def make_grain_gevals(grains=GRAINS, threshold: float = 0.8, judge=None) -> List[GEval]:
    return [grain_geval(g, threshold, judge) for g in grains]


def intent_geval(threshold: float = 0.7, judge=None) -> GEval:
    """Query-level: does the SQL answer the user's question? (your original 'intent match')"""
    return GEval(
        name="Intent Match",
        evaluation_steps=[
            "Infer the user's analytical intent from 'input'.",
            "Judge whether 'actual output' answers that intent, using 'expected output' as reference.",
            "Penalize wrong grain, missing filters, or answering a different question.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=INTENT_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def query_match_geval(threshold: float = 0.8, judge=None) -> GEval:
    """Query-level: would predicted SQL match gold's result? (your original 'SQL match')"""
    return GEval(
        name="Query Match",
        evaluation_steps=[
            "Treat 'actual output' and 'expected output' as SQL queries.",
            "Judge whether they would return the same result set for any valid data state.",
            "Account for equivalent rewrites; penalize INNER vs LEFT, missing DISTINCT, different grain.",
        ],
        evaluation_params=[P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=QUERY_MATCH_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def make_full_grain_metrics(grains=GRAINS, threshold: float = 0.8,
                            judge=None) -> List[GEval]:
    """All grain GEvals PLUS intent match and query match — the complete set as
    separate report rows."""
    return (make_grain_gevals(grains, threshold, judge)
            + [intent_geval(threshold, judge), query_match_geval(threshold, judge)])
