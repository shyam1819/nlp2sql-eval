"""
Verdict metrics for NL2SQL: Intent Match + Query Match.

These two decide pass/fail and are kept SEPARATE (two floors, not a blend):
  - Intent Match : does the SQL answer the user's actual question?
  - Query Match  : would the SQL return the same result set as the gold query?

Each is a GEval with its own banded rubric. Run the judge at temperature=0
for stability. (Grains stay diagnostic elsewhere; grounding/execution are
separate axes and are intentionally NOT included here.)
"""
from __future__ import annotations
from typing import List, Optional

from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import SingleTurnParams as P


# --- rubrics (0-10, non-overlapping bands) ---------------------------------
INTENT_RUBRIC = [
    Rubric(score_range=(0, 2),   expected_outcome="Answers a different question than the one asked."),
    Rubric(score_range=(3, 6),   expected_outcome="Addresses the question but misses part of the intent (wrong grain, a missing condition)."),
    Rubric(score_range=(7, 9),   expected_outcome="Answers the user's intent with only a minor gap."),
    Rubric(score_range=(10, 10), expected_outcome="Fully and precisely answers the user's analytical intent."),
]

QUERY_MATCH_RUBRIC = [
    Rubric(score_range=(0, 2),   expected_outcome="Would return a clearly different result set than gold."),
    Rubric(score_range=(3, 6),   expected_outcome="Partially overlapping results; notable differences (grain, join type, dupes)."),
    Rubric(score_range=(7, 9),   expected_outcome="Same results except a minor edge case."),
    Rubric(score_range=(10, 10), expected_outcome="Semantically equivalent to gold for any valid data state."),
]


# --- metrics ---------------------------------------------------------------
def intent_match_metric(threshold: float = 0.7, judge=None) -> GEval:
    return GEval(
        name="Intent Match",
        evaluation_steps=[
            "Read 'input' and identify what the user asked for: the subject/entities, the measure or output wanted, the grain to break it down by, and any scope or conditions stated.",
            "Use 'expected output' ONLY to clarify what the question is asking for — not as a template the SQL must copy.",
            "Judge whether 'actual output' is aimed at that same question: does it return the requested measure, at the requested grain, over the requested scope?",
            "Do NOT judge whether its RESULTS match the gold query — that is Query Match's job. Here, judge only whether the RIGHT QUESTION is being answered.",
            "Score low only when it answers a different question: wrong subject, wrong measure, a grain the user did not ask for, or a stated condition ignored.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=INTENT_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def query_match_metric(threshold: float = 0.8, judge=None) -> GEval:
    return GEval(
        name="Query Match",
        evaluation_steps=[
            "Compare 'actual output' and 'expected output' purely as two SQL queries; ignore the user question here.",
            "Decide whether they would return the SAME rows and columns for EVERY possible database state.",
            "IGNORE non-material differences — ones that CANNOT change the result: alias or CTE names, column or predicate ordering, JOIN vs equivalent IN/EXISTS, CTE vs subquery, whitespace/formatting.",
            "PENALIZE material differences — ones that CAN change the result: INNER vs LEFT/OUTER join, missing or extra DISTINCT, a different GROUP BY grain, different filter bounds (> vs >=), a different aggregate or selected-column set, or a LIMIT/ORDER BY that changes which rows are returned.",
            "Give full marks only if the two queries are provably equivalent for any data; otherwise score down in proportion to how much the result sets could diverge.",
        ],
        evaluation_params=[P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=QUERY_MATCH_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def make_verdict_metrics(intent_threshold: float = 0.7,
                         query_threshold: float = 0.8,
                         judge=None) -> List[GEval]:
    """Both verdict metrics as separate report rows (separate floors, no blend)."""
    return [intent_match_metric(intent_threshold, judge),
            query_match_metric(query_threshold, judge)]


# Optional: temperature-0 judge for stable scores. Swap provider as needed.
def make_judge(model: str = "gpt-4o", temperature: float = 0.0,
               api_key: Optional[str] = None):
    from deepeval.models import GPTModel
    return GPTModel(model=model, temperature=temperature, api_key=api_key)


def run_verdict_eval(test_cases, judge=None,
                     intent_threshold: float = 0.7,
                     query_threshold: float = 0.8,
                     print_results: bool = True,
                     run_async: bool = True):
    """Build the two verdict metrics, run evaluate(), and return the
    EvaluationResult. `test_cases` is a list of LLMTestCase.

    Because both metrics must pass for a case to pass, each TestResult.success
    is already the conjunctive verdict (intent AND query_match) — no blending.
    """
    from deepeval import evaluate
    from deepeval.evaluate.configs import DisplayConfig, AsyncConfig

    metrics = make_verdict_metrics(intent_threshold, query_threshold, judge)
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        display_config=DisplayConfig(print_results=print_results),
        async_config=AsyncConfig(run_async=run_async),
    )


def verdict_summary(result) -> List[dict]:
    """Flatten an EvaluationResult into per-case rows with each floor's score
    plus the conjunctive pass/fail. Handy for building your own report table."""
    rows = []
    for tr in result.test_results:
        scores = {m.name: {"score": m.score, "pass": m.success, "reason": m.reason}
                  for m in (tr.metrics_data or [])}
        intent = scores.get("Intent Match", {})
        query = scores.get("Query Match", {})
        rows.append({
            "index": tr.index,
            "input": tr.input,
            "intent_score": intent.get("score"),
            "intent_pass": intent.get("pass"),
            "query_match_score": query.get("score"),
            "query_match_pass": query.get("pass"),
            "verdict_pass": bool(tr.success),   # intent AND query_match
            "intent_reason": intent.get("reason"),
            "query_match_reason": query.get("reason"),
        })
    return rows
