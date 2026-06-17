"""The OrganizationalState aggregate — an in-memory projection of the company.

This is a plain data structure (no ORM, no I/O). The builder fills it from
memory; reasoning reads it. Keeping it pure makes it trivially testable and
serializable into a snapshot.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class CommitmentView:
    id: uuid.UUID
    description: str
    status: str
    due_at: datetime | None
    promised_at: datetime | None
    age_days: float | None
    counterparty: str | None


@dataclass(slots=True)
class RelationshipView:
    id: uuid.UUID
    person: str | None
    kind: str
    last_contact_at: datetime | None
    days_since_contact: float | None
    health: float


@dataclass(slots=True)
class ProjectView:
    id: uuid.UUID
    name: str
    status: str
    days_since_activity: float | None


@dataclass(slots=True)
class CommitmentState:
    open: list[CommitmentView] = field(default_factory=list)
    overdue: list[CommitmentView] = field(default_factory=list)

    @property
    def total_open(self) -> int:
        return len(self.open) + len(self.overdue)


@dataclass(slots=True)
class RelationshipState:
    relationships: list[RelationshipView] = field(default_factory=list)

    def going_cold(self, threshold_days: float = 7.0) -> list[RelationshipView]:
        return [
            r
            for r in self.relationships
            if r.days_since_contact is not None and r.days_since_contact >= threshold_days
        ]


@dataclass(slots=True)
class ProjectState:
    projects: list[ProjectView] = field(default_factory=list)

    def stalled(self, threshold_days: float = 10.0) -> list[ProjectView]:
        return [
            p
            for p in self.projects
            if p.days_since_activity is not None and p.days_since_activity >= threshold_days
        ]


@dataclass(slots=True)
class PeopleState:
    total: int = 0


@dataclass(slots=True)
class OrganizationalState:
    """The central aggregate. Built every morning; read by every engine."""

    org_id: uuid.UUID
    built_at: datetime
    people: PeopleState = field(default_factory=PeopleState)
    commitments: CommitmentState = field(default_factory=CommitmentState)
    relationships: RelationshipState = field(default_factory=RelationshipState)
    projects: ProjectState = field(default_factory=ProjectState)

    def summary(self) -> dict[str, Any]:
        return {
            "org_id": str(self.org_id),
            "built_at": self.built_at.isoformat(),
            "people_total": self.people.total,
            "commitments_open": self.commitments.total_open,
            "commitments_overdue": len(self.commitments.overdue),
            "relationships_going_cold": len(self.relationships.going_cold()),
            "projects_stalled": len(self.projects.stalled()),
        }
