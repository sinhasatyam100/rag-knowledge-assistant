"""
streamlit_app.py — RAG Knowledge Assistant UI
Deploy on Streamlit Cloud, pointing to your GCP FastAPI backend.

SSE event types emitted by /ask/stream (from main.py):
    {"type": "cache_hit", "similarity": "..."}
    {"type": "sources",   "sources": [...]}
    {"type": "token",     "content": "..."}
    {"type": "done",      "full_response": "...", "from_cache": bool}
    {"type": "error",     "message": "..."}

/health response shape (from main.py):
    {"status": "healthy", "documents_indexed": N, "cache": {...}, "timestamp": "..."}

Deploy:
    1. Push this file to your GitHub repo root
    2. Go to share.streamlit.io → New app → select this file
    3. Add secret: API_BASE = "https://rag-api-308467052823.asia-south1.run.app"
"""

import streamlit as st
import requests
import json
import time

import os

# ── Config ────────────────────────────────────────────────────────────
# Priority:
# 1. Streamlit secrets (Streamlit Cloud production deployment)
# 2. Environment variable (docker-compose local dev: API_BASE=http://api:8000)
# 3. Hardcoded GCP URL (fallback for quick local testing with uvicorn)
try:
    st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
    API_BASE = st.secrets["API_BASE"].rstrip("/")
except (KeyError, FileNotFoundError):
    API_BASE = os.environ.get(
        "API_BASE",
        "https://rag-api-308467052823.asia-south1.run.app"
    ).rstrip("/")


st.markdown("""
<style>
.source-chip {
    display: inline-block; background: #f0f2f6;
    border: 1px solid #d0d3db; border-radius: 6px;
    padding: 3px 10px; font-size: 12px; color: #444;
    margin: 3px 3px 0 0; word-break: break-all;
}
.cache-badge {
    display: inline-block; background: #fff3cd;
    border: 1px solid #ffc107; border-radius: 6px;
    padding: 2px 8px; font-size: 11px; color: #856404; margin-bottom: 4px;
}
.metric-card {
    background: #f8f9fa; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 6px; font-size: 13px;
}
.healthy   { color: #1a7a4a; font-weight: 600; }
.unhealthy { color: #c0392b; font-weight: 600; }
.warming   { color: #b7621a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for _k, _v in [("messages", []), ("health", None), ("health_ts", 0)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────

def fetch_health(force: bool = False):
    now = time.time()
    if force or (now - st.session_state.health_ts > 30):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=15)
            st.session_state.health = r.json() if r.ok else None
        except requests.exceptions.Timeout:
            st.session_state.health = {"_cold_start": True}
        except Exception:
            st.session_state.health = None
        st.session_state.health_ts = now
    return st.session_state.health


def stream_answer(question: str, top_k: int = 5, history: list = None):
    """POST to /ask/stream and yield parsed SSE event dicts."""
    url = f"{API_BASE}/ask/stream"
    # Send the last 10 messages of history (5 turns) for context.
    # Exclude the message we just added (the current question) — it's in 'question'.
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in (history or [])[-10:]
        if m["role"] in ("user", "assistant")
    ]
    try:
        with requests.post(url, json={
            "question": question,
            "top_k":    top_k,
            "history":  history_payload,
        }, stream=True, timeout=90) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if not data_str:
                    continue
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.Timeout:
        yield {"type": "error", "message": "Timed out — API may be cold-starting. Wait a few seconds and retry."}
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "message": f"Cannot reach {API_BASE}. Check that the GCP service is running."}
    except requests.exceptions.HTTPError as e:
        yield {"type": "error", "message": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}


def ingest_files(files):
    file_tuples = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
    try:
        r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=90)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"error": str(e)}


def ingest_confluence(base_url, username, api_token, space_key, max_pages, mock):
    try:
        r = requests.post(f"{API_BASE}/ingest/confluence", timeout=120, json={
            "base_url": base_url, "username": username, "api_token": api_token,
            "space_key": space_key, "max_pages": max_pages, "mock": mock,
        })
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"error": str(e)}


def ingest_jira(base_url, username, api_token, project_key, max_issues, mock):
    try:
        r = requests.post(f"{API_BASE}/ingest/jira", timeout=120, json={
            "base_url": base_url, "username": username, "api_token": api_token,
            "project_key": project_key, "max_issues": max_issues, "mock": mock,
        })
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:300]}"}
    except Exception as e:
        return {"error": str(e)}


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption("Internal Knowledge Base")

    health = fetch_health()
    if health is None:
        st.markdown('<span class="unhealthy">● API unreachable</span>', unsafe_allow_html=True)
    elif health.get("_cold_start"):
        st.markdown('<span class="warming">● API warming up…</span>', unsafe_allow_html=True)
        st.caption("Cloud Run cold start — refresh in ~10s")
    else:
        doc_count = health.get("documents_indexed", "?")
        cache = health.get("cache", {})
        st.markdown('<span class="healthy">● API healthy</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">📄 <b>{doc_count}</b> docs indexed</div>',
                    unsafe_allow_html=True)
        if cache:
            hit_rate = cache.get("hit_rate_pct", cache.get("hit_rate", 0))
            entries = cache.get("entries", "?")
            st.markdown(
                f'<div class="metric-card">⚡ Cache: <b>{hit_rate}%</b> hit rate · {entries} entries</div>',
                unsafe_allow_html=True)

    if st.button("↻ Refresh status", use_container_width=True):
        fetch_health(force=True)
        st.rerun()

    st.divider()
    top_k = st.slider("Sources to retrieve (top-k)", 2, 10, 5)
    st.divider()

    st.subheader("Ingest Documents")
    source = st.radio("Source", ["📄 Files", "🔵 Confluence", "🟠 JIRA"],
                      label_visibility="collapsed")

    # ── File upload ───────────────────────────────────────────────────
    if source == "📄 Files":
        uploaded = st.file_uploader("Upload .txt or .pdf", accept_multiple_files=True,
                                    type=["txt", "pdf"])
        if st.button("Ingest Files", use_container_width=True, disabled=not uploaded):
            with st.spinner("Ingesting…"):
                result = ingest_files(uploaded)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✓ {result.get('total_chunks_added', 0)} chunks added")
                for r in result.get("results", []):
                    icon = "✓" if r["status"] == "success" else "✗"
                    st.caption(f"{icon} {r['file']}: {r.get('chunks_added', r.get('reason', ''))}")
                fetch_health(force=True)

    # ── Confluence ────────────────────────────────────────────────────
    elif source == "🔵 Confluence":
        use_mock_cf = st.toggle("Use mock data", value=True, key="cf_mock")
        with st.form("confluence_form"):
            cf_url   = st.text_input("Base URL",
                                     value="https://mock.atlassian.net" if use_mock_cf else "",
                                     placeholder="https://yourorg.atlassian.net",
                                     disabled=use_mock_cf)
            cf_user  = st.text_input("Email",
                                     value="demo@acme.com" if use_mock_cf else "",
                                     disabled=use_mock_cf)
            cf_token = st.text_input("API token",
                                     value="mock" if use_mock_cf else "",
                                     type="password", disabled=use_mock_cf)
            cf_space = st.text_input(
                "Space key",
                value="MOCK" if use_mock_cf else "",
                placeholder="ENG",
                help=(
                    "Enter the Confluence space key only, e.g.ENG.\n\n"
                    "If you paste a full URL like https://yoursite/wiki/spaces//pages/...\n"
                    "the connector will extract  automatically."
                ),
            )
            cf_max   = st.number_input("Max pages", 1, 500, 50)
            cf_go    = st.form_submit_button("Ingest Confluence", use_container_width=True)

        if cf_go:
            space = (cf_space.strip().upper()) or ("MOCK" if use_mock_cf else "")
            if not space:
                st.error("Space key required.")
            else:
                with st.spinner(f"Fetching {'mock ' if use_mock_cf else ''}Confluence…"):
                    result = ingest_confluence(
                        cf_url or "https://mock.atlassian.net",
                        cf_user or "demo", cf_token or "mock",
                        space, cf_max, use_mock_cf,
                    )
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✓ {result.get('pages_fetched',0)} pages → {result.get('chunks_added',0)} chunks")
                    fetch_health(force=True)

    # ── JIRA ──────────────────────────────────────────────────────────
    elif source == "🟠 JIRA":
        use_mock_j = st.toggle("Use mock data", value=True, key="j_mock")
        with st.form("jira_form"):
            j_url   = st.text_input("Base URL",
                                    value="https://mock.atlassian.net" if use_mock_j else "",
                                    placeholder="https://yourorg.atlassian.net",
                                    disabled=use_mock_j)
            j_user  = st.text_input("Email",
                                    value="demo@acme.com" if use_mock_j else "",
                                    disabled=use_mock_j)
            j_token = st.text_input("API token",
                                    value="mock" if use_mock_j else "",
                                    type="password", disabled=use_mock_j)
            j_proj  = st.text_input("Project key",
                                    value="MOCK" if use_mock_j else "",
                                    placeholder="QA")
            j_max   = st.number_input("Max issues", 1, 1000, 100)
            j_go    = st.form_submit_button("Ingest JIRA", use_container_width=True)

        if j_go:
            project = (j_proj.strip().upper()) or ("MOCK" if use_mock_j else "")
            if not project:
                st.error("Project key required.")
            else:
                with st.spinner(f"Fetching {'mock ' if use_mock_j else ''}JIRA…"):
                    result = ingest_jira(
                        j_url or "https://mock.atlassian.net",
                        j_user or "demo", j_token or "mock",
                        project, j_max, use_mock_j,
                    )
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✓ {result.get('issues_fetched',0)} issues → {result.get('chunks_added',0)} chunks")
                    fetch_health(force=True)

    st.divider()
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat ─────────────────────────────────────────────────────────
st.header("Ask your knowledge base")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("from_cache"):
                st.markdown('<span class="cache-badge">⚡ From semantic cache</span>',
                            unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                    for s in msg["sources"]:
                        src = s.get("source", "unknown")
                        preview = s.get("preview", "")
                        st.markdown(f'<span class="source-chip">📄 {src}</span>',
                                    unsafe_allow_html=True)
                        if preview:
                            st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))

if question := st.chat_input("Ask a question about your documents…"):
    cid      = st.session_state.active_chat_id
    chat_obj = st.session_state.chats[cid]

        # Auto-name the chat from the first question
    if chat_obj["name"] == "New chat" and not chat_obj["messages"]:
        chat_obj["name"] = (question[:35] + "…") if len(question) > 35 else question

    # Append user message
    chat_obj["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build prior_messages AFTER appending current question,
    # then slice off the last item (current question) so history
    # only contains previous turns.
    prior_messages = chat_obj["messages"][:-1]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources     = []
        from_cache  = False
        error_msg   = None

        # Pass all messages except the one we just appended (current question)
        prior_messages = chat_obj["messages"][:-1]
        for event in stream_answer(question, top_k=top_k, history=prior_messages):
            etype = event.get("type")
            if etype == "cache_hit":
                from_cache = True
            elif etype == "sources":
                sources = event.get("sources", [])
            elif etype == "token":
                full_answer += event.get("content", "")
                placeholder.markdown(full_answer + "▌")
            elif etype == "done":
                full_answer = event.get("full_response", full_answer)
                from_cache  = event.get("from_cache", from_cache)
                placeholder.markdown(full_answer)
            elif etype == "error":
                error_msg = event.get("message", "Unknown error")
                placeholder.error(f"⚠️ {error_msg}")
                break

        if from_cache:
            st.markdown('<span class="cache-badge">⚡ From semantic cache</span>',
                        unsafe_allow_html=True)
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    src = s.get("source", "unknown")
                    preview = s.get("preview", "")
                    st.markdown(f'<span class="source-chip">📄 {src}</span>',
                                unsafe_allow_html=True)
                    if preview:
                        st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))

    if full_answer and not error_msg:
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "from_cache": from_cache,
        })
