"""
connectors/confluence.py

Fetches pages from Confluence via REST API and returns LangChain Documents.

Supports Atlassian Cloud and self-hosted Confluence Server.
The API URL is built as: {base_url}/rest/api/content

For Atlassian Cloud, pass base_url including /wiki:
    https://yourorg.atlassian.net/wiki

For self-hosted Server with a context path, include it:
    https://yourorg.example.com/confluence

You may paste any full page URL into either base_url or space_key —
both fields are parsed automatically and the correct values extracted.

Mock mode (mock=True) returns built-in demo data with no credentials needed.
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

_HTML_TAG   = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s{2,}")

def _strip_html(html: str) -> str:
    text = _HTML_TAG.sub(" ", html or "")
    return _WHITESPACE.sub(" ", text).strip()


# ── URL parsing helpers ───────────────────────────────────────────────

_TERMINAL = frozenset({
    "spaces", "pages", "display", "rest", "api",
    "content", "viewpage.action", "search",
})
_CONTEXT  = frozenset({"confluence", "wiki", "jira"})


def _extract_base_url(raw: str) -> str:
    """
    Extract the base URL (scheme + host + context path) from any input.

    Examples:
        "https://yourorg.atlassian.net/wiki/spaces/ENG/pages/..."
            -> "https://yourorg.atlassian.net/wiki"

        "https://yourorg.example.com/confluence/spaces/ABC/pages/..."
            -> "https://yourorg.example.com/confluence"

        "https://yourorg.atlassian.net"
            -> "https://yourorg.atlassian.net"
    """
    raw = raw.strip().rstrip("/")
    if not raw.startswith("http"):
        raw = "https://" + raw

    m = re.match(r"(https?://[^/]+)(.*)", raw)
    if not m:
        return raw

    host = m.group(1)
    path = m.group(2)

    kept = []
    for seg in path.split("/"):
        if not seg:
            continue
        if seg.lower() in _TERMINAL or seg.isdigit():
            break
        kept.append(seg)

    context = "/" + "/".join(kept) if kept else ""
    return host + context


def _extract_space_key(raw: str) -> str:
    """
    Extract the Confluence space key from any input.

    Examples:
        "https://yourorg.atlassian.net/wiki/spaces/ENG/pages/..." -> "ENG"
        "ENG/pages/123/Page+Title"                                 -> "ENG"
        "ENG"                                                      -> "ENG"
    """
    raw  = raw.strip()
    path = re.sub(r"^https?://[^/]+", "", raw) if raw.startswith("http") else raw

    segments = [s for s in path.split("/") if s and "%" not in s]
    _non_key = _TERMINAL | _CONTEXT

    for i, seg in enumerate(segments):
        if seg.lower() == "spaces" and i + 1 < len(segments):
            candidate = segments[i + 1]
            if not candidate.isdigit() and candidate.lower() not in _non_key:
                return candidate.upper()

    for seg in segments:
        if seg.lower() in _non_key or seg.isdigit():
            continue
        return seg.upper()

    return re.sub(r"[^A-Z0-9_]", "", raw.upper())[:20] or raw.upper()


# ── Mock data ─────────────────────────────────────────────────────────

MOCK_PAGES = [
    {
        "id": "1001",
        "title": "Engineering Onboarding Guide",
        "space": "ENG",
        "author": "alice@example.com",
        "last_modified": "2024-11-01T09:00:00Z",
        "body": """
        Welcome to the Engineering team. This guide covers your first 30 days.

        Development Environment Setup:
        Clone the monorepo: git clone git@github.com:acme/platform.git
        Install dependencies: make install
        Run tests: make test
        All services run locally via docker-compose up.

        Branching Strategy:
        We use trunk-based development. Feature branches should be short-lived (under 2 days).
        Branch naming: feature/<ticket-id>-short-description
        All PRs require 2 approvals and passing CI before merge.

        Deployment Process:
        Staging deploys automatically on merge to main.
        Production deployments are triggered manually via the deploy pipeline.
        Announce releases in the #deployments channel.

        Code Review Guidelines:
        Reviewers should respond within 1 business day.
        Nitpicks should be prefixed with nit: to signal they are optional.
        """,
    },
    {
        "id": "1002",
        "title": "QA Process and Test Strategy",
        "space": "ENG",
        "author": "bob@example.com",
        "last_modified": "2024-10-15T14:30:00Z",
        "body": """
        Our QA process ensures every feature ships with confidence.

        Test Levels:
        Unit tests: written by developers, run in CI on every commit, coverage gate 80%.
        Integration tests: verify service boundaries, run in staging.
        E2E tests: Selenium suite runs nightly against staging.
        Performance tests: Locust load tests run for every release candidate.

        Bug Classification:
        P0: production outage, fix within 2 hours.
        P1: critical feature broken, fix within 24 hours.
        P2: non-critical, fix in current sprint.
        P3: cosmetic, fix when convenient.

        Release Criteria:
        Zero P0 or P1 bugs in staging before production release.
        All automated test suites passing.
        Manual regression completed for changed user flows.
        """,
    },
    {
        "id": "1003",
        "title": "Incident Response Runbook",
        "space": "OPS",
        "author": "carol@example.com",
        "last_modified": "2024-12-01T08:00:00Z",
        "body": """
        When an incident occurs, follow this runbook to minimise time-to-resolution.

        Step 1 - Detect and Declare:
        Any engineer can declare an incident by posting in #incidents.
        Severity levels: SEV1 (total outage), SEV2 (degraded), SEV3 (minor).

        Step 2 - Assign Roles:
        Incident Commander: owns communication and drives resolution.
        Technical Lead: investigates root cause and coordinates fixes.
        Comms Lead: updates status page and notifies stakeholders.

        Step 3 - Investigate:
        Check dashboards for error rate spikes and latency increases.
        Review recent deployments in the deploy log.

        Step 4 - Resolve and Review:
        Schedule a blameless postmortem within 48 hours.
        Document root cause, timeline, and action items.
        All action items must have an owner and due date.
        """,
    },
    {
        "id": "1004",
        "title": "API Design Standards",
        "space": "ENG",
        "author": "dave@example.com",
        "last_modified": "2024-09-20T11:00:00Z",
        "body": """
        All internal and external APIs must follow these standards.

        RESTful Conventions:
        Use plural nouns for resource names: /users not /user.
        Status codes: 200 success, 201 created, 204 no-content, 400 bad request,
        401 unauthenticated, 403 forbidden, 404 not found, 500 server error.

        Versioning:
        Version via URL path: /v1/users.
        Breaking changes require a new major version.
        Old versions supported for 12 months after new version release.

        Pagination:
        Use cursor-based pagination for large collections.
        Response must include: data[], next_cursor, total_count.
        Default page size 20, maximum 100.

        Authentication:
        All endpoints require Bearer token authentication.
        Tokens expire after 1 hour. Rate limits: 1000 requests per minute per API key.
        """,
    },
    {
        "id": "1005",
        "title": "Data Retention and Privacy Policy",
        "space": "LEGAL",
        "author": "eve@example.com",
        "last_modified": "2024-08-10T16:00:00Z",
        "body": """
        This policy governs how the organisation stores, uses, and deletes customer data.

        Retention Periods:
        Active account data: retained for the lifetime of the account.
        Inactive accounts: data deleted 3 years after last login.
        Financial records: retained 7 years per regulatory requirement.
        Log data: 90 days hot storage, 1 year cold storage.

        Customer Rights (GDPR):
        Right to access: fulfilled within 30 days.
        Right to erasure: data deleted within 30 days of verified request.
        Right to portability: data exported in JSON or CSV format.

        Data Processing:
        No customer data is sold to third parties.
        Sub-processors must meet ISO 27001 or SOC 2 Type II.
        """,
    },
]


# ── Connector ─────────────────────────────────────────────────────────

class ConfluenceConnector:
    """
    Fetches Confluence pages and returns LangChain Documents.

    Usage — real instance:
        connector = ConfluenceConnector(
            base_url="https://yourorg.atlassian.net/wiki",
            username="you@example.com",
            api_token="your_api_token",
        )
        docs = connector.fetch_space("ENG", max_pages=100)

    Usage — mock mode:
        connector = ConfluenceConnector.mock()
        docs = connector.fetch_space("MOCK")
    """

    def __init__(
        self,
        base_url:  str,
        username:  str,
        api_token: str,
        _mock:     bool = False,
    ):
        self.base_url = _extract_base_url(base_url)
        self.auth     = HTTPBasicAuth(username, api_token)
        self._mock    = _mock
        self._session = requests.Session()
        self._session.auth = self.auth
        self._session.headers.update({
            "Accept":       "application/json",
            "Content-Type": "application/json",
        })

    @classmethod
    def mock(cls) -> "ConfluenceConnector":
        return cls(
            base_url="https://mock.example.com",
            username="mock",
            api_token="mock",
            _mock=True,
        )

    def fetch_space(
        self,
        space_key: str,
        max_pages: int = 100,
        since:     Optional[datetime] = None,
    ) -> list[Document]:
        """
        Fetch pages from a Confluence space.
        space_key accepts a bare key (ENG) or any full page URL.
        """
        if self._mock:
            return self._mock_fetch(space_key, max_pages, since)
        return self._real_fetch(_extract_space_key(space_key), max_pages, since)

    def _real_fetch(
        self,
        space_key: str,
        max_pages: int,
        since:     Optional[datetime],
    ) -> list[Document]:
        docs    = []
        start   = 0
        limit   = min(50, max_pages)
        api_url = f"{self.base_url}/rest/api/content"

        while len(docs) < max_pages:
            params = {
                "spaceKey": space_key,
                "type":     "page",
                "status":   "current",
                "expand":   "body.storage,version,space",
                "start":    start,
                "limit":    limit,
            }
            try:
                resp = self._session.get(api_url, params=params, timeout=30)
            except requests.exceptions.SSLError as e:
                raise RuntimeError(f"SSL error connecting to {self.base_url}: {e}") from e
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(
                    f"Cannot connect to {self.base_url}. "
                    f"Ensure the host is reachable: {e}"
                ) from e
            except requests.exceptions.Timeout:
                raise RuntimeError(f"Request to {api_url} timed out after 30 seconds.")

            if resp.status_code == 401:
                raise RuntimeError("401 Unauthorized. Check your username and API token.")
            if resp.status_code == 403:
                raise RuntimeError(
                    f"403 Forbidden for space '{space_key}'. "
                    "Check that your account has read access."
                )
            if resp.status_code == 404:
                raise RuntimeError(
                    f"404 Not Found at {api_url}. "
                    f"Check the base URL is correct: {self.base_url}"
                )
            if resp.status_code == 503:
                raise RuntimeError(
                    f"503 Service Unavailable at {api_url}. "
                    "The server is unreachable from this machine."
                )

            resp.raise_for_status()

            try:
                data = resp.json()
            except Exception:
                raise RuntimeError(
                    f"Non-JSON response (status {resp.status_code}): {resp.text[:300]}"
                )

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                doc = self._page_to_document(page, space_key)
                if doc is None:
                    continue
                if since:
                    last_mod_str = page.get("version", {}).get("when", "")
                    if last_mod_str:
                        try:
                            last_mod = datetime.fromisoformat(
                                last_mod_str.replace("Z", "+00:00")
                            )
                            if last_mod <= since.replace(tzinfo=timezone.utc):
                                continue
                        except ValueError:
                            pass
                docs.append(doc)
                if len(docs) >= max_pages:
                    break

            if not data.get("_links", {}).get("next") or len(results) < limit:
                break
            start += limit
            time.sleep(0.1)

        return docs

    def _page_to_document(self, page: dict, space_key: str) -> Optional[Document]:
        try:
            page_id       = page["id"]
            title         = page.get("title", "Untitled")
            raw_html      = page.get("body", {}).get("storage", {}).get("value", "")
            text          = _strip_html(raw_html)
            if not text.strip():
                return None
            last_modified = page.get("version", {}).get("when", "")
            author        = page.get("version", {}).get("by", {}).get("displayName", "unknown")
            page_url      = f"{self.base_url}/spaces/{space_key}/pages/{page_id}"
            return Document(
                page_content=f"{title}\n\n{text}",
                metadata={
                    "source":        page_url,
                    "title":         title,
                    "space_key":     space_key,
                    "page_id":       page_id,
                    "last_modified": last_modified,
                    "author":        author,
                    "connector":     "confluence",
                },
            )
        except Exception:
            return None

    def _mock_fetch(
        self,
        space_key: str,
        max_pages: int,
        since:     Optional[datetime],
    ) -> list[Document]:
        docs       = []
        key        = space_key.upper()
        candidates = MOCK_PAGES if key == "MOCK" else (
            [p for p in MOCK_PAGES if p["space"] == key] or MOCK_PAGES
        )
        for page in candidates[:max_pages]:
            if since:
                last_mod = datetime.fromisoformat(
                    page["last_modified"].replace("Z", "+00:00")
                )
                if last_mod <= since.replace(tzinfo=timezone.utc):
                    continue
            docs.append(Document(
                page_content=f"{page['title']}\n\n{page['body'].strip()}",
                metadata={
                    "source":        f"https://mock.example.com/spaces/{page['space']}/pages/{page['id']}",
                    "title":         page["title"],
                    "space_key":     page["space"],
                    "page_id":       page["id"],
                    "last_modified": page["last_modified"],
                    "author":        page["author"],
                    "connector":     "confluence",
                },
            ))
        return docs
