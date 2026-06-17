"""build_organizational_state — read memory, project the current company state."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from navis.memory.models import (
    Commitment,
    CommitmentStatus,
    Person,
    Project,
    Relationship,
)
from navis.state.aggregate import (
    CommitmentState,
    CommitmentView,
    OrganizationalState,
    PeopleState,
    ProjectState,
    ProjectView,
    RelationshipState,
    RelationshipView,
)


def _days_since(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 86400.0


async def build_organizational_state(db: AsyncSession, org_id: uuid.UUID) -> OrganizationalState:
    now = datetime.now(timezone.utc)

    # People count
    people_total = await db.scalar(
        select(func.count(Person.id)).where(Person.organization_id == org_id)
    ) or 0

    # Commitments
    commitments = (
        await db.scalars(
            select(Commitment).where(
                Commitment.organization_id == org_id,
                Commitment.status.in_([CommitmentStatus.OPEN, CommitmentStatus.OVERDUE]),
            )
        )
    ).all()

    commitment_state = CommitmentState()
    for c in commitments:
        view = CommitmentView(
            id=c.id,
            description=c.description,
            status=c.status.value,
            due_at=c.due_at,
            promised_at=c.promised_at or c.created_at,
            age_days=_days_since(c.promised_at or c.created_at, now),
            counterparty=str(c.counterparty_person_id) if c.counterparty_person_id else None,
        )
        is_overdue = c.status == CommitmentStatus.OVERDUE or (
            c.due_at is not None and _days_since(c.due_at, now) and _days_since(c.due_at, now) > 0
        )
        (commitment_state.overdue if is_overdue else commitment_state.open).append(view)

    # Relationships
    relationships = (
        await db.scalars(
            select(Relationship).where(
                Relationship.organization_id == org_id,
                Relationship.is_active.is_(True),
            )
        )
    ).all()
    rel_state = RelationshipState(
        relationships=[
            RelationshipView(
                id=r.id,
                person=str(r.person_id) if r.person_id else None,
                kind=r.kind.value,
                last_contact_at=r.last_contact_at,
                days_since_contact=_days_since(r.last_contact_at, now),
                health=r.health,
            )
            for r in relationships
        ]
    )

    # Projects
    projects = (
        await db.scalars(select(Project).where(Project.organization_id == org_id))
    ).all()
    proj_state = ProjectState(
        projects=[
            ProjectView(
                id=p.id,
                name=p.name,
                status=p.status.value,
                days_since_activity=_days_since(p.last_activity_at, now),
            )
            for p in projects
        ]
    )

    return OrganizationalState(
        org_id=org_id,
        built_at=now,
        people=PeopleState(total=int(people_total)),
        commitments=commitment_state,
        relationships=rel_state,
        projects=proj_state,
    )
