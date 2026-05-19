"""
connectors/jira.py

Fetches JIRA issues via REST API and returns LangChain Documents.
Works with Atlassian Cloud (api_token auth).

Usage — real instance:
    connector = JiraConnector(
        base_url="https://yourorg.atlassian.net",
        username="you@email.com",
        api_token="your_api_token",
    )
    docs = connector.fetch_project("QA", max_issues=200)

Usage — mock mode (no real JIRA needed):
    connector = JiraConnector.mock()
    docs = connector.fetch_project("MOCK", max_issues=50)
"""

from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Optional
import requests
from requests.auth import HTTPBasicAuth
from langchain_core.documents import Document


# ── Mock data ─────────────────────────────────────────────────────────
# Realistic JIRA issues covering common enterprise scenarios.
# Combine summary + description + comments into one document per issue
# so the full context is retrievable in a single chunk.

MOCK_ISSUES = [
    {
        "key": "ENG-101",
        "project": "ENG",
        "summary": "API response time exceeds 2s for /search endpoint under load",
        "status": "In Progress",
        "priority": "High",
        "assignee": "alice@acme.com",
        "reporter": "bob@acme.com",
        "created": "2024-10-01T10:00:00Z",
        "updated": "2024-11-15T14:30:00Z",
        "description": """
        The /search endpoint response time degrades significantly above 50 concurrent users.
        P99 latency spikes to 4.2 seconds, breaching our 2s SLA.

        Steps to reproduce:
        1. Run Locust load test: locust -f tests/load/search_load.py --users 100
        2. Observe latency in Datadog dashboard (search-api-latency)
        3. P99 climbs above 2s at ~50 concurrent users

        Root cause investigation:
        - Database query in SearchService.findByKeyword() is missing an index on the 'tags' column
        - Full table scan on 2.3M rows at each request
        - Query plan confirmed with EXPLAIN ANALYZE

        Fix:
        - Add composite index: CREATE INDEX idx_tags_created ON documents(tags, created_at)
        - Query optimisation in SearchService to use the new index
        - Expected improvement: P99 < 500ms at 200 concurrent users
        """,
        "comments": [
            {"author": "carol@acme.com", "body": "Confirmed in staging. EXPLAIN ANALYZE shows seq scan on documents table. Index creation in progress."},
            {"author": "alice@acme.com", "body": "Index created in staging. Latency P99 now 380ms at 200 users. Rolling to prod in next release."},
        ],
    },
    {
        "key": "QA-55",
        "project": "QA",
        "summary": "Login flow breaks on iOS 17 Safari — blank screen after OAuth redirect",
        "status": "Open",
        "priority": "Critical",
        "assignee": "dave@acme.com",
        "reporter": "eve@acme.com",
        "created": "2024-11-20T09:00:00Z",
        "updated": "2024-11-21T11:00:00Z",
        "description": """
        Users on iOS 17 with Safari are getting a blank white screen after completing
        Google OAuth login. The redirect back to the app completes (URL changes to /dashboard)
        but the page remains blank.

        Affected: iOS 17.0, 17.1, 17.2 on Safari. Chrome on iOS unaffected. Android unaffected.

        Steps to reproduce:
        1. Open app in Safari on iOS 17 device
        2. Tap "Sign in with Google"
        3. Complete Google auth
        4. Observe blank screen at /dashboard

        Console errors (from Safari web inspector):
        - "SecurityError: Blocked a frame with origin 'https://app.acme.com' from accessing a cross-origin frame"
        - Related to Safari's ITP (Intelligent Tracking Prevention) blocking postMessage

        Expected behaviour: User lands on dashboard with session active.
        """,
        "comments": [
            {"author": "frank@acme.com", "body": "This is a Safari ITP issue. The OAuth popup is being blocked from posting the auth token back to the parent frame. Fix: switch from popup OAuth to redirect flow."},
            {"author": "dave@acme.com", "body": "Spike in progress. Using redirect flow (no popup) resolves the issue in local testing on iOS 17 simulator."},
        ],
    },
    {
        "key": "OPS-22",
        "project": "OPS",
        "summary": "Cloud Run service cold start latency — 8-12s on first request after scale-to-zero",
        "status": "Resolved",
        "priority": "Medium",
        "assignee": "grace@acme.com",
        "reporter": "henry@acme.com",
        "created": "2024-09-10T08:00:00Z",
        "updated": "2024-10-05T16:00:00Z",
        "description": """
        After periods of inactivity, the first request to our Cloud Run service takes 8-12 seconds
        to respond due to cold start. This causes timeouts for users who hit the app after off-hours.

        Observed behaviour:
        - Idle period > 15 minutes triggers scale-to-zero
        - Next request: 8-12s for container to start + load ML model
        - Subsequent requests: normal (<500ms)

        The ML model loading (SentenceTransformer all-MiniLM-L6-v2) accounts for ~6s of cold start.

        Resolution options considered:
        1. Min instances = 1 (always warm, $45/month extra cost)
        2. Startup probe with /health endpoint (reduce timeout, not latency)
        3. Model baked into Docker image (saves ~2s download, not 6s load time)
        4. Cloud Run CPU always allocated (not boost-only)
        """,
        "comments": [
            {"author": "grace@acme.com", "body": "Set min-instances=1 during business hours (8am-8pm IST) via Cloud Scheduler. Off-hours cold start acceptable. Cost: ~$22/month."},
            {"author": "henry@acme.com", "body": "Closing. Min-instances scheduler deployed. Cold starts during business hours eliminated. Users no longer hitting timeouts."},
        ],
    },
    {
        "key": "ENG-87",
        "project": "ENG",
        "summary": "Implement rate limiting on /api/v1 endpoints — 100 req/min per API key",
        "status": "Done",
        "priority": "High",
        "assignee": "irene@acme.com",
        "reporter": "james@acme.com",
        "created": "2024-08-05T11:00:00Z",
        "updated": "2024-09-01T09:00:00Z",
        "description": """
        External API partners are occasionally hammering endpoints, causing latency spikes
        for other tenants. We need per-API-key rate limiting at the gateway level.

        Requirements:
        - 100 requests/minute per API key (configurable per partner)
        - Burst allowance: 20 requests in first second
        - 429 Too Many Requests response with Retry-After header
        - Rate limit counters stored in Redis (already available in infra)
        - Whitelist for internal services (no limit)
        - Rate limit metrics in Datadog: rl.hits, rl.rejected per key

        Implementation:
        - FastAPI middleware using Redis sliding window counter
        - Key: rate_limit:{api_key}:{minute_bucket}
        - Increment on each request, expire after 2 minutes
        - Check against limit before routing to handler
        """,
        "comments": [
            {"author": "irene@acme.com", "body": "Implemented Redis sliding window rate limiter as FastAPI middleware. Tests passing. Deployed to staging."},
            {"author": "james@acme.com", "body": "Verified in staging. Partner A hitting limit correctly getting 429s. Internal service whitelisted and unaffected. Merged."},
        ],
    },
    {
        "key": "QA-71",
        "project": "QA",
        "summary": "Selenium E2E suite flakiness — 15% of test runs fail on CI due to timing issues",
        "status": "In Progress",
        "priority": "Medium",
        "assignee": "kate@acme.com",
        "reporter": "leo@acme.com",
        "created": "2024-11-01T10:00:00Z",
        "updated": "2024-11-18T14:00:00Z",
        "description": """
        Our Selenium E2E test suite on CI (GitHub Actions + ChromeDriver) has a 15% flakiness rate.
        Tests pass consistently locally but fail intermittently in CI.

        Failure patterns identified:
        1. StaleElementReferenceException on dynamic content (React re-renders between locate and click)
        2. ElementNotInteractableException on dropdowns (element visible but not yet clickable)
        3. TimeoutException on API-dependent assertions (slow CI network, API takes >3s)

        Current approach: explicit waits (WebDriverWait) with 10s timeout — insufficient for slow CI.

        Proposed fixes:
        1. Increase WebDriverWait to 20s for API-dependent steps
        2. Replace findElement with retry wrapper that catches StaleElementReferenceException
        3. Add network idle detection before asserting API-driven content
        4. Tag known flaky tests with @flaky annotation and run them 3x with pass-on-first-pass logic
        """,
        "comments": [
            {"author": "kate@acme.com", "body": "Retry wrapper implemented and deployed. Flakiness dropped from 15% to 4% in first week. StaleElementReference issues resolved."},
            {"author": "kate@acme.com", "body": "Remaining 4% flakiness traced to two specific tests with external API dependencies. Adding mock for those APIs in CI environment."},
        ],
    },
]


class JiraConnector:
    """Fetches JIRA issues and returns LangChain Documents."""

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

    @classmethod
    def mock(cls) -> "JiraConnector":
        """Returns a connector that uses built-in mock data. No credentials needed."""
        return cls(base_url="http://mock", username="mock", api_token="mock", _mock=True)

    # ── Public ────────────────────────────────────────────────────────

    def fetch_project(
        self,
        project_key: str,
        max_issues: int = 100,
        since: Optional[datetime] = None,
    ) -> list[Document]:
        """
        Fetch all issues from a JIRA project.

        Args:
            project_key: JIRA project key (e.g. "QA", "ENG").
            max_issues: Maximum number of issues to fetch.
            since: If provided, only fetch issues updated after this datetime.

        Returns:
            List of LangChain Documents ready for chunking and indexing.
        """
        if self._mock:
            return self._mock_fetch(project_key, max_issues, since)
        return self._real_fetch(project_key, max_issues, since)

    # ── Real JIRA REST API ────────────────────────────────────────────

    def _real_fetch(
        self,
        project_key: str,
        max_issues: int,
        since: Optional[datetime],
    ) -> list[Document]:
        docs = []
        start_at = 0
        page_size = min(50, max_issues)

        # Build JQL
        jql = f"project = {project_key} ORDER BY updated DESC"
        if since:
            since_str = since.strftime("%Y-%m-%d %H:%M")
            jql = f"project = {project_key} AND updated >= '{since_str}' ORDER BY updated DESC"

        while len(docs) < max_issues:
            url = f"{self.base_url}/rest/api/3/search"
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": page_size,
                "fields": "summary,description,status,priority,assignee,reporter,created,updated,comment",
            }
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            issues = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                doc = self._issue_to_document(issue)
                if doc:
                    docs.append(doc)
                if len(docs) >= max_issues:
                    break

            total = data.get("total", 0)
            start_at += len(issues)
            if start_at >= total or len(issues) < page_size:
                break

            time.sleep(0.1)

        return docs

    def _issue_to_document(self, issue: dict) -> Optional[Document]:
        """Convert a raw JIRA API issue dict to a LangChain Document."""
        try:
            key = issue["key"]
            fields = issue.get("fields", {})

            summary = fields.get("summary", "No summary")
            status = fields.get("status", {}).get("name", "Unknown")
            priority = fields.get("priority", {}).get("name", "Unknown")
            assignee = (fields.get("assignee") or {}).get("displayName", "Unassigned")
            reporter = (fields.get("reporter") or {}).get("displayName", "Unknown")
            updated = fields.get("updated", "")

            # Description — JIRA API v3 returns Atlassian Document Format (ADF)
            # We extract plain text from the ADF content blocks
            description = self._extract_adf_text(fields.get("description") or {})

            # Comments
            comments_data = fields.get("comment", {}).get("comments", [])
            comments_text = "\n".join([
                f"Comment by {c.get('author', {}).get('displayName', 'unknown')}: "
                f"{self._extract_adf_text(c.get('body', {}))}"
                for c in comments_data
            ])

            # Combine all text — summary + description + comments
            full_text = f"{summary}\n\n{description}"
            if comments_text:
                full_text += f"\n\nComments:\n{comments_text}"

            if not full_text.strip():
                return None

            return Document(
                page_content=full_text,
                metadata={
                    "source": f"{self.base_url}/browse/{key}",
                    "issue_key": key,
                    "summary": summary,
                    "status": status,
                    "priority": priority,
                    "assignee": assignee,
                    "reporter": reporter,
                    "last_modified": updated,
                    "connector": "jira",
                },
            )
        except Exception:
            return None

    def _extract_adf_text(self, adf: dict) -> str:
        """
        Recursively extract plain text from Atlassian Document Format (ADF).
        ADF is a nested JSON structure used by JIRA API v3.
        """
        if not adf:
            return ""
        texts = []
        if adf.get("type") == "text":
            texts.append(adf.get("text", ""))
        for child in adf.get("content", []):
            texts.append(self._extract_adf_text(child))
        return " ".join(t for t in texts if t).strip()

    # ── Mock mode ─────────────────────────────────────────────────────

    def _mock_fetch(
        self,
        project_key: str,
        max_issues: int,
        since: Optional[datetime],
    ) -> list[Document]:
        """Return built-in mock issues, optionally filtered by project."""
        docs = []
        candidates = MOCK_ISSUES if project_key == "MOCK" else [
            i for i in MOCK_ISSUES if i["project"] == project_key
        ] or MOCK_ISSUES  # Fall back to all if project not found

        for issue in candidates[:max_issues]:
            if since:
                updated = datetime.fromisoformat(issue["updated"].replace("Z", "+00:00"))
                if updated <= since.replace(tzinfo=timezone.utc):
                    continue

            comments_text = "\n".join([
                f"Comment by {c['author']}: {c['body'].strip()}"
                for c in issue.get("comments", [])
            ])

            full_text = f"{issue['summary']}\n\n{issue['description'].strip()}"
            if comments_text:
                full_text += f"\n\nComments:\n{comments_text}"

            docs.append(Document(
                page_content=full_text,
                metadata={
                    "source": f"https://mock.atlassian.net/browse/{issue['key']}",
                    "issue_key": issue["key"],
                    "summary": issue["summary"],
                    "status": issue["status"],
                    "priority": issue["priority"],
                    "assignee": issue["assignee"],
                    "reporter": issue["reporter"],
                    "last_modified": issue["updated"],
                    "connector": "jira",
                },
            ))
        return docs
