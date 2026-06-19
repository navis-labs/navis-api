"""Organization V1 — "what company is this, and what can it do?"

Owns profile, lifecycle (TRIAL/ACTIVE/SUSPENDED/CHURNED), onboarding (a monotonic
stage funnel), entitlements, plans/billing, server-side feature flags, activation
(TTFI), and metrics. Does NOT own users/sessions/roles/permissions/api-keys/SSO/
SCIM — those stay in Identity.

`OrganizationService.can_use(org_id, capability)` is the single capability authority
every business domain consults (via `organization.guards.require_capability`).
Organization depends only on platform + identity; business domains depend on it.
"""
