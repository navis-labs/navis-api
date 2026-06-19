"""Intelligence — what has this organization learned about itself?

The final layer. It consumes Learning ONLY (learning_events / learning_aggregates /
learning_patterns) and projects it into append-only IntelligenceInsights — immutable
FACTS the organization has learned about itself (signal/decision/team/customer/
organizational/adaptation), each fully traceable to its supporting learning events.

Not recommendations. Not actions. Not scores. Not risk. Deterministic and rebuildable:
drop the table, re-project Learning, get identical insights.
"""
