"""Orchestration — workflow control only. NO business logic.

Orchestrators sequence steps across domains (onboarding, sync, the morning
brief). Each step's logic lives in its domain; orchestration just says
"step A, then B, then C" and owns scheduling/retries.
"""
