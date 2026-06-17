"""Execution — Layer 6. Human-approved actions.

A decision becomes an Action. Actions are NEVER auto-fired: they require human
approval (the product is human-in-the-loop). Commands: send_email, create_task,
schedule_meeting, create_workflow.

    Decision -> Action(requested) -> [human approves] -> Action(executed) -> Outcome
"""
