# nlp2sql-eval

## Intent Match

"Read 'input' and identify what the user asked for: the subject/entities, the measure or output wanted, the grain to break it down by, and any scope or conditions stated.",
"Use 'expected output' ONLY to clarify what the question is asking for — not as a template the SQL must copy.",
"Judge whether 'actual output' is aimed at that same question: does it return the requested measure, at the requested grain, over the requested scope?",
"Do NOT judge whether its RESULTS match the gold query — that is Query Match's job. Here, judge only whether the RIGHT QUESTION is being answered.",
"Score low only when it answers a different question: wrong subject, wrong measure, a grain the user did not ask for, or a stated condition ignored.",

## Query Match

"Compare 'actual output' and 'expected output' purely as two SQL queries; ignore the user question here.",
"Decide whether they would return the SAME rows and columns for EVERY possible database state.",
"IGNORE non-material differences — ones that CANNOT change the result: alias or CTE names, column or predicate ordering, JOIN vs equivalent IN/EXISTS, CTE vs subquery, whitespace/formatting.",
"PENALIZE material differences — ones that CAN change the result: INNER vs LEFT/OUTER join, missing or extra DISTINCT, a different GROUP BY grain, different filter bounds (> vs >=), a different aggregate or selected-column set, or a LIMIT/ORDER BY that changes which rows are returned.",
"Give full marks only if the two queries are provably equivalent for any data; otherwise score down in proportion to how much the result sets could diverge.",
