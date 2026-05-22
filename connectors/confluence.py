"""
connectors/confluence.py

Fetches pages from Confluence via REST API and returns LangChain Documents.
Works with Atlassian Cloud (api_token) and Confluence Server (basic auth).

Usage — real instance:
    connector = ConfluenceConnector(
        base_url="https://yourorg.atlassian.net",
        username="you@email.com",
        api_token="your_api_token",
    )
    docs = connector.fetch_space("ENG", max_pages=100)

Usage — mock mode (no real Confluence needed):
    connector = ConfluenceConnector.mock()
    docs = connector.fetch_space("MOCK", max_pages=50)
"""

from __future__ import annotations
import re
import time
from datetime import datetime, timezone
from typing import Optional
import requests
from requests.auth import HTTPBasicAuth
from langchain_core.documents import Document


# ── HTML stripping ────────────────────────────────────────────────────
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s{2,}")

def _strip_html(html: str) -> str:
    text = _HTML_TAG.sub(" ", html or "")
    return _WHITESPACE.sub(" ", text).strip()

def _parse_confluence_url(raw: str):
    raw = raw.strip()
 
    # Strip scheme + host to get the path portion
    if raw.startswith("http"):
        path = re.sub(r"^https?://[^/]+", "", raw)
    else:
        path = raw
 
    # Split path into segments, dropping empty strings and URL-encoded noise
    segments = [s for s in path.split("/") if s and "%" not in s]
 
    # Known Confluence context path prefixes (the bit before /spaces/ or /rest/)
    CONTEXT_PREFIXES = {"confluence", "wiki", "display"}
    # Segment names that are never a space key
    NON_KEY = {"spaces", "pages", "display", "rest", "api", "content", "space"}
 
    context_path = None
    space_key = None
 
    # Walk segments looking for the pattern: [context?] -> "spaces" -> KEY
    for i, seg in enumerate(segments):
        if seg.lower() == "spaces" and i + 1 < len(segments):
            candidate = segments[i + 1]
            if not candidate.isdigit() and candidate.lower() not in NON_KEY:
                space_key = candidate.upper()
                # Context path is everything before "spaces"
                prefix_segs = segments[:i]
                # Only keep segments that look like context paths
                ctx_parts = [s for s in prefix_segs
                             if s.lower() in CONTEXT_PREFIXES
                             or (not s.isdigit() and s.lower() not in NON_KEY)]
                context_path = "/" + "/".join(ctx_parts) if ctx_parts else ""
                return context_path, space_key
 
    # No /spaces/ pattern found — try to extract a bare space key
    for seg in segments:
        if seg.lower() in NON_KEY or seg.isdigit() or seg.lower() in CONTEXT_PREFIXES:
            continue
        space_key = seg.upper()
        break
 
    if space_key is None:
        # Last resort: strip non-alphanumeric chars
        space_key = re.sub(r"[^A-Z0-9_]", "", path.upper())[:20] or path.upper()
 
    return None, space_key  # context_path unknown — will be probed
 
 
def _sanitise_space_key(raw: str) -> str:
    """Return just the space key from whatever the user pastes."""
    _, key = _parse_confluence_url(raw)
    return key


def _detect_api_prefix(session: requests.Session, base_url: str,
                       hint_context: Optional[str] = None) -> str:
    """
    Return the correct REST API prefix for this Confluence instance.
 
    hint_context: if we parsed a context path from the user's URL (e.g.
    "/confluence"), try that first before the generic fallbacks.
    """
    candidates = []
 
    # If we have a hint from the user's URL, try it first
    if hint_context is not None:
        candidates.append(f"{hint_context}/rest/api")
 
    # Standard fallbacks
    candidates += ["/wiki/rest/api", "/rest/api", "/confluence/rest/api"]
 
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
 
    for prefix in ordered:
        try:
            url = f"{base_url}{prefix}/space"
            r = session.get(url, params={"limit": 1}, timeout=10)
            # 200 = works, 401/403 = right path, wrong/no credentials
            if r.status_code in (200, 401, 403):
                return prefix
        except requests.exceptions.RequestException:
            continue
 
    # Nothing responded — return the hint or the Cloud default
    return (f"{hint_context}/rest/api" if hint_context else "/wiki/rest/api")

# ── Mock data ─────────────────────────────────────────────────────────
MOCK_PAGES = [
    {
        "id": "1001",
        "title": "Engineering Onboarding Guide",
        "space": "ENG",
        "author": "alice@acme.com",
        "last_modified": "2024-11-01T09:00:00Z",
        "body": """
        Welcome to the Engineering team. This guide covers your first 30 days.

        Development Environment Setup:
        Clone the monorepo from GitHub: git clone git@github.com:acme/platform.git
        Install dependencies with: make install
        Run tests with: make test
        All services run locally via docker-compose up.

        Branching Strategy:
        We use trunk-based development. Feature branches should be short-lived (< 2 days).
        Branch naming: feature/<ticket-id>-short-description
        All PRs require 2 approvals and passing CI before merge.

        Deployment Process:
        Staging deploys automatically on merge to main.
        Production deployments are triggered manually via the deploy pipeline in Jenkins.
        Use the #deployments Slack channel to announce releases.

        Code Review Guidelines:
        Reviewers should respond within 1 business day.
        Focus feedback on correctness, security, and maintainability.
        Nitpicks should be prefixed with "nit:" to signal they are optional.
        """,
    },
    {
        "id": "1002",
        "title": "QA Process and Test Strategy",
        "space": "ENG",
        "author": "bob@acme.com",
        "last_modified": "2024-10-15T14:30:00Z",
        "body": """
        Our QA process ensures every feature ships with confidence.

        Test Levels:
        Unit tests: written by developers, run in CI on every commit. Coverage gate: 80%.
        Integration tests: verify service boundaries, run in staging environment.
        E2E tests: Selenium/Playwright suite, runs nightly against staging.
        Performance tests: Locust load tests run every release candidate.

        Bug Classification:
        P0 — production outage, fix within 2 hours, all hands.
        P1 — critical feature broken, fix within 24 hours.
        P2 — non-critical issue, fix in current sprint.
        P3 — cosmetic/minor, fix when convenient.

        Release Criteria:
        Zero P0 or P1 bugs in staging before production release.
        All automated test suites passing.
        Manual regression completed for changed user flows.

        Test Data Management:
        Use the test data factory (tests/factories/) to generate realistic fixtures.
        Never use production data in tests. Anonymised snapshots available on request.
        """,
    },
    {
        "id": "1003",
        "title": "Incident Response Runbook",
        "space": "OPS",
        "author": "carol@acme.com",
        "last_modified": "2024-12-01T08:00:00Z",
        "body": """
        When an incident occurs, follow this runbook to minimise time-to-resolution.

        Step 1 — Detect and Declare:
        Any engineer can declare an incident by posting in #incidents.
        Use /incident start <severity> <description> in Slack.
        Severity levels: SEV1 (total outage), SEV2 (degraded), SEV3 (minor).

        Step 2 — Assign Roles:
        Incident Commander: owns communication and drives resolution.
        Technical Lead: investigates root cause and coordinates fixes.
        Comms Lead: updates status page and notifies stakeholders.

        Step 3 — Investigate:
        Check Datadog dashboards for error rate spikes and latency increases.
        Review recent deployments in the deploy log.
        Check CloudWatch logs for error patterns.

        Step 4 — Resolve and Review:
        After resolving, schedule a blameless postmortem within 48 hours.
        Document root cause, timeline, and action items in Confluence.
        All action items must have an owner and due date.
        """,
    },
    {
        "id": "1004",
        "title": "API Design Standards",
        "space": "ENG",
        "author": "dave@acme.com",
        "last_modified": "2024-09-20T11:00:00Z",
        "body": """
        All internal and external APIs must follow these standards.

        RESTful Conventions:
        Use plural nouns for resource names: /users not /user.
        HTTP verbs: GET (read), POST (create), PUT (replace), PATCH (update), DELETE (remove).
        Return 200 for success, 201 for created, 204 for no-content, 400 for bad request,
        401 for unauthenticated, 403 for forbidden, 404 for not found, 500 for server error.

        Versioning:
        Version via URL path: /v1/users. Do not version via headers.
        Breaking changes require a new major version.
        Old versions supported for 12 months after new version release.

        Pagination:
        Use cursor-based pagination for large collections.
        Response must include: data[], next_cursor, total_count.
        Default page size: 20. Maximum: 100.

        Authentication:
        All endpoints require Bearer token authentication.
        Tokens expire after 1 hour. Refresh tokens valid for 30 days.
        Rate limits: 1000 requests/minute per API key.
        """,
    },
    {
        "id": "1005",
        "title": "Data Retention and Privacy Policy",
        "space": "LEGAL",
        "author": "eve@acme.com",
        "last_modified": "2024-08-10T16:00:00Z",
        "body": """
        This policy governs how ACME stores, uses, and deletes customer data.

        Data Categories:
        PII (Personally Identifiable Information): name, email, phone, address.
        Usage data: clickstreams, feature usage, session recordings.
        Financial data: payment methods, transaction history.
        Health data: only collected for health-product customers, requires explicit consent.

        Retention Periods:
        Active account data: retained for the lifetime of the account.
        Inactive accounts: data deleted 3 years after last login.
        Financial records: retained 7 years per regulatory requirement.
        Support tickets: retained 2 years.
        Log data: retained 90 days in hot storage, 1 year in cold storage.

        Customer Rights (GDPR/DPDP):
        Right to access: customer can request a copy of their data within 30 days.
        Right to erasure: data deleted within 30 days of verified request.
        Right to portability: data exported in machine-readable format (JSON/CSV).

        Data Processing:
        No customer data is sold to third parties.
        Sub-processors must sign DPA and meet ISO 27001 or SOC 2 Type II.
        Data transferred outside India requires Standard Contractual Clauses.
        """,
    },
]


# ── Connector ─────────────────────────────────────────────────────────

class ConfluenceConnector:
    """Fetches Confluence pages and returns LangChain Documents."""

    def __init__(
        self,
        base_url: str,
        username: str,
        api_token: str,
        _mock: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth = HTTPBasicAuth(username, api_token)
        self._mock = _mock
        self._session = requests.Session()
        self._session.auth = self.auth
        self._session.headers.update({"Accept": "application/json"})
        # Detected lazily on first real fetch
        self._api_prefix: Optional[str] = None

    @classmethod
    def mock(cls) -> "ConfluenceConnector":
        """Returns a connector that uses built-in mock data. No credentials needed."""
        return cls(base_url="http://mock", username="mock", api_token="mock", _mock=True)

    def _get_api_prefix(self) -> str:
        if self._api_prefix is None:
            self._api_prefix = _detect_api_prefix(self._session, self.base_url)
        return self._api_prefix

    # ── Public ────────────────────────────────────────────────────────

    def fetch_space(
        self,
        space_key: str,
        max_pages: int = 100,
        since: Optional[datetime] = None,
    ) -> list[Document]:
        """
        Fetch all pages in a Confluence space.

        space_key can be a bare key ("ENG"), a full URL, or a partial path —
        _sanitise_space_key() will extract the correct key automatically.
        """
        if self._mock:
            return self._mock_fetch(space_key, max_pages, since)

        clean_key = _sanitise_space_key(space_key)
        return self._real_fetch(clean_key, max_pages, since)

    # ── Real Confluence REST API ──────────────────────────────────────

    def _real_fetch(
        self,
        space_key: str,
        max_pages: int,
        since: Optional[datetime],
    ) -> list[Document]:
        docs = []
        start = 0
        limit = min(50, max_pages)
        prefix = self._get_api_prefix()

        while len(docs) < max_pages:
            url = f"{self.base_url}{prefix}/content"
            params = {
                "spaceKey": space_key,
                "type": "page",
                "status": "current",
                "expand": "body.storage,version,metadata.labels",
                "start": start,
                "limit": limit,
            }
            try:
                resp = self._session.get(url, params=params, timeout=30)
            except requests.exceptions.SSLError as e:
                raise RuntimeError(
                    f"SSL error connecting to {self.base_url}. "
                    "If this is a self-signed cert server, the container needs "
                    "the cert added. Original error: {e}"
                ) from e
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to {self.base_url}. "
                    "Check that the host is reachable from the Docker container "
                    "(it may be behind a VPN or firewall). Original error: {e}"
                ) from e

            if resp.status_code == 401:
                raise RuntimeError(
                    "Confluence returned 401 Unauthorized. "
                    "Check your email and API token."
                )
            if resp.status_code == 403:
                raise RuntimeError(
                    f"Confluence returned 403 Forbidden for space '{space_key}'. "
                    "Your account may not have permission to read this space."
                )
            if resp.status_code == 503:
                raise RuntimeError(
                    f"Confluence returned 503 Service Unavailable at {url}. "
                    "The server may be behind a VPN — make sure Docker has network "
                    "access to the Confluence host. If running locally, confirm you "
                    "are on the corporate VPN and the container can reach the host."
                )

            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                doc = self._page_to_document(page, space_key, prefix)
                if doc is None:
                    continue
                if since:
                    last_mod_str = page.get("version", {}).get("when", "")
                    if last_mod_str:
                        last_mod = datetime.fromisoformat(last_mod_str.replace("Z", "+00:00"))
                        if last_mod <= since.replace(tzinfo=timezone.utc):
                            continue
                docs.append(doc)
                if len(docs) >= max_pages:
                    break

            next_link = data.get("_links", {}).get("next")
            if not next_link or len(results) < limit:
                break
            start += limit
            time.sleep(0.1)

        return docs

    def _page_to_document(self, page: dict, space_key: str, prefix: str) -> Optional[Document]:
        """Convert a raw Confluence API page dict to a LangChain Document."""
        try:
            page_id = page["id"]
            title = page.get("title", "Untitled")
            raw_html = page.get("body", {}).get("storage", {}).get("value", "")
            text = _strip_html(raw_html)

            if not text.strip():
                return None

            last_modified = page.get("version", {}).get("when", "")
            author = page.get("version", {}).get("by", {}).get("displayName", "unknown")

            # Build the human-readable page URL depending on server type
            if prefix == "/wiki/rest/api":
                page_url = f"{self.base_url}/wiki/spaces/{space_key}/pages/{page_id}"
            else:
                page_url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"

            return Document(
                page_content=f"{title}\n\n{text}",
                metadata={
                    "source": page_url,
                    "title": title,
                    "space_key": space_key,
                    "page_id": page_id,
                    "last_modified": last_modified,
                    "author": author,
                    "connector": "confluence",
                },
            )
        except Exception:
            return None

    # ── Mock mode ─────────────────────────────────────────────────────

    def _mock_fetch(
        self,
        space_key: str,
        max_pages: int,
        since: Optional[datetime],
    ) -> list[Document]:
        """Return built-in mock pages, optionally filtered by space and since."""
        docs = []
        candidates = MOCK_PAGES if space_key == "MOCK" else [
            p for p in MOCK_PAGES if p["space"] == space_key
        ] or MOCK_PAGES

        for page in candidates[:max_pages]:
            if since:
                last_mod = datetime.fromisoformat(page["last_modified"].replace("Z", "+00:00"))
                if last_mod <= since.replace(tzinfo=timezone.utc):
                    continue
            docs.append(Document(
                page_content=f"{page['title']}\n\n{page['body'].strip()}",
                metadata={
                    "source": f"https://mock.atlassian.net/wiki/spaces/{page['space']}/pages/{page['id']}",
                    "title": page["title"],
                    "space_key": page["space"],
                    "page_id": page["id"],
                    "last_modified": page["last_modified"],
                    "author": page["author"],
                    "connector": "confluence",
                },
            ))
        return docs
