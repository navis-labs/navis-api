"""
Identity — users, memberships, roles, permissions,
authentication, authorization, sessions,
API keys, service accounts, SSO and SCIM.

Identity owns:

- Who a principal is
- Which organizations they belong to
- What permissions they have

Identity does NOT own organization lifecycle,
billing, onboarding, entitlements, or feature flags.

Every domain is scoped by organization_id.
Identity enforces that tenancy boundary.
"""