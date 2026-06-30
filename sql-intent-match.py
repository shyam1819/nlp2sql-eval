"""
LLM-as-judge evaluation framework for NL2SQL (DeepEval), Layer 3 of the cascade.

Covers what deterministic structure can't:
  - intent alignment      : does the SQL answer the user's real question?
  - semantic equivalence  : are predicted & gold SQL equivalent despite syntax?
                            (this is what rescues grain-match FALSE NEGATIVES)
  - schema grounding      : did it invent tables/columns not in the schema?

Anti-fluctuation design (your original problem):
  1. evaluation_steps instead of free-text criteria  (reproducible CoT)
  2. Rubric bands                                     (no mid-score clustering)
  3. temperature=0 judge model                        (deterministic decoding)
  4. StableGEval self-consistency wrapper             (median of N samples)
  5. SQLCorrectnessCascade                            (LLM fires only in the
     gray zone -> fewer calls => less cost AND less variance)

Pairs with the deterministic grains in sql_grain_eval.py.
"""
from __future__ import annotations
from statistics import median, pstdev
from typing import List, Optional

from deepeval.metrics import BaseMetric, GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams as P

# deterministic grain layer (Layer 2) for the cascade
from sql_grain_eval import grain_analysis, weighted_score, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Judge model: temperature 0 is lever #3 against fluctuation.
# Swap GPTModel for AnthropicModel / GeminiModel / AzureOpenAIModel / LocalModel.
# ---------------------------------------------------------------------------
def make_judge(model: str = "gpt-4o", temperature: float = 0.0,
               api_key: Optional[str] = None):
    from deepeval.models import GPTModel
    return GPTModel(model=model, temperature=temperature, api_key=api_key)


_CORRECTNESS_RUBRIC = [
    Rubric(score_range=(0, 2), expected_outcome="Wrong: different results or wrong intent."),
    Rubric(score_range=(3, 6), expected_outcome="Partially right: minor logic/column issues."),
    Rubric(score_range=(7, 9), expected_outcome="Right but stylistically different from gold."),
    Rubric(score_range=(10, 10), expected_outcome="Fully correct and intent-complete."),
]


# ---------------------------------------------------------------------------
# GEval-based LLM metrics (Layer 3). Pass judge=make_judge() to fix the model.
# ---------------------------------------------------------------------------
def intent_alignment_metric(threshold: float = 0.7, judge=None) -> GEval:
    """Reference-based: does the SQL satisfy the user's question intent?"""
    return GEval(
        name="Intent Alignment",
        evaluation_steps=[
            "Read the user question in 'input' and infer the analytical intent.",
            "Determine whether 'actual output' SQL would answer that intent.",
            "Compare against 'expected output' for the columns/grouping the user asked for.",
            "Penalize wrong aggregation level, missing filters, or answering a different question.",
            "Do NOT penalize stylistic differences that don't change the result.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=_CORRECTNESS_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def semantic_equivalence_metric(threshold: float = 0.8, judge=None) -> GEval:
    """Are predicted & gold SQL semantically equivalent? Rescues grain false negatives."""
    return GEval(
        name="Semantic Equivalence",
        evaluation_steps=[
            "Treat 'actual output' and 'expected output' as SQL queries.",
            "Decide if they return the SAME result set for any valid database state.",
            "Account for equivalent rewrites: predicate order, JOIN vs IN/EXISTS, "
            "CTE vs subquery, alias names, column order.",
            "Return high only if results are provably equivalent; penalize subtle "
            "differences (e.g. INNER vs LEFT join, missing DISTINCT, different GROUP BY).",
        ],
        evaluation_params=[P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
        rubric=_CORRECTNESS_RUBRIC,
        threshold=threshold,
        model=judge,
    )


def schema_grounding_metric(threshold: float = 0.9, judge=None) -> GEval:
    """Did the SQL reference only tables/columns present in the schema (CONTEXT)?"""
    return GEval(
        name="Schema Grounding",
        evaluation_steps=[
            "The 'context' contains the available schema (tables and columns).",
            "Check every table and column referenced in 'actual output' exists in context.",
            "Heavily penalize any hallucinated table or column not in the schema.",
        ],
        evaluation_params=[P.ACTUAL_OUTPUT, P.CONTEXT],
        rubric=[
            Rubric(score_range=(0, 4), expected_outcome="References objects not in schema."),
            Rubric(score_range=(5, 9), expected_outcome="Mostly grounded, minor issues."),
            Rubric(score_range=(10, 10), expected_outcome="Every reference exists in schema."),
        ],
        threshold=threshold,
        model=judge,
    )


def intent_only_metric(threshold: float = 0.7, judge=None) -> GEval:
    """Reference-FREE variant for when you have no gold SQL."""
    return GEval(
        name="Intent (reference-free)",
        evaluation_steps=[
            "Infer the user's analytical intent from 'input'.",
            "Judge whether 'actual output' SQL plausibly and completely answers it.",
            "Penalize wrong grain, missing filters, or unanswered parts of the question.",
        ],
        evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
        rubric=_CORRECTNESS_RUBRIC,
        threshold=threshold,
        model=judge,
    )


# ---------------------------------------------------------------------------
# Anti-fluctuation lever #4: self-consistency wrapper.
# Runs an inner GEval N times, reports the MEDIAN and the spread.
# ---------------------------------------------------------------------------
class StableGEval(BaseMetric):
    def __init__(self, inner: BaseMetric, samples: int = 5, threshold: float = 0.7):
        self.inner = inner
        self.samples = samples
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            scores, reasons = [], []
            for _ in range(self.samples):
                scores.append(self.inner.measure(test_case))
                reasons.append(getattr(self.inner, "reason", None))
            self.score = median(scores)
            spread = pstdev(scores) if len(scores) > 1 else 0.0
            self.reason = (f"median={self.score:.2f} over {self.samples} runs "
                           f"(stdev={spread:.3f}, range={min(scores):.2f}-{max(scores):.2f}). "
                           f"Sample reason: {reasons[-1]}")
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e); self.success = False; raise

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
        return f"Stable[{getattr(self.inner, '__name__', 'GEval')}] x{self.samples}"


# ---------------------------------------------------------------------------
# The cascade: deterministic grains decide the confident cases; the LLM judge
# is invoked ONLY in the uncertain middle band. Minimizes cost + variance.
# ---------------------------------------------------------------------------
class SQLCorrectnessCascade(BaseMetric):
    def __init__(self, judge=None, dialect: str = "databricks",
                 schema: Optional[dict] = None,
                 high_conf: float = 0.99, low_conf: float = 0.40,
                 threshold: float = 0.8, weights: dict = DEFAULT_WEIGHTS):
        self.judge = judge
        self.dialect = dialect
        self.schema = schema
        self.high_conf = high_conf      # grain score above this => confident PASS, no LLM
        self.low_conf = low_conf        # grain score below this => confident FAIL, no LLM
        self.threshold = threshold
        self.weights = weights
        self.async_mode = False
        self.include_reason = True
        self.error = None
        self.score = 0.0
        self.reason = None
        self.success = False
        self._equiv = None

    def _equiv_metric(self):
        if self._equiv is None:
            self._equiv = semantic_equivalence_metric(threshold=self.threshold, judge=self.judge)
        return self._equiv

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            results = grain_analysis(test_case.actual_output, test_case.expected_output,
                                     self.dialect, self.schema)
            g = weighted_score(results, self.weights)
            if g >= self.high_conf:
                self.score, self.reason = g, f"Grains fully match (={g:.2f}); LLM skipped."
            elif g <= self.low_conf:
                self.score, self.reason = g, f"Grains strongly mismatch (={g:.2f}); LLM skipped."
            else:
                llm = self._equiv_metric().measure(test_case)   # only here do we pay for an LLM
                self.score = llm
                self.reason = (f"Gray zone (grain={g:.2f}); LLM equivalence={llm:.2f}. "
                               f"{getattr(self._equiv, 'reason', '')}")
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e); self.success = False; raise

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
        return "SQL Correctness (grain+LLM cascade)"
