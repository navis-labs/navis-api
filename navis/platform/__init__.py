"""Platform — shared infrastructure only.

Config, database, cache, queue, events, storage, monitoring, security.

GOLDEN RULE: never put business/domain logic here. If a module knows what a
`Commitment` or a `DailyBrief` is, it does not belong in platform.
"""
