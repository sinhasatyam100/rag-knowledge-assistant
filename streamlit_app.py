"""
streamlit_app.py — RAG Knowledge Assistant UI

SSE event types from /ask/stream:
    {"type": "cache_hit"}
    {"type": "sources",  "sources": [...]}
    {"type": "token",    "content": "..."}
    {"type": "done",     "full_response": "...", "from_cache": bool}
    {"type": "error",    "message": "..."}

Deploy:
    1. Push to GitHub repo root
    2. share.streamlit.io → New app → select this file
    3. Add secret: API_BASE = "https://rag-api-308467052823.asia-south1.run.app"
"""

import os
import time
import json
import uuid
from datetime import datetime

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────
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
/* existing styles ... */
.stChatInput {            
    position: fixed;
    bottom: 10px;
    left: calc(var(--sidebar-width) + 20px);;
    right: calc(var(--sidebar-width) + 20px);
}

/* Add bottom padding to chat area so messages don't hide behind input */
section.main > div {
    padding-bottom: 80px;
}

/* Right panel header alignment */
[data-testid="column"]:last-child {
    border-left: 1px solid #f0f0f0;
    padding-left: 16px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────

def _new_chat_id() -> str:
    return str(uuid.uuid4())[:8]

def _new_chat(name: str = "New chat") -> dict:
    return {
        "name":       name,
        "messages":   [],
        "created_at": datetime.now().strftime("%d %b, %H:%M"),
    }

# Initialise all session state keys on first load
if "chats" not in st.session_state:
    first_id = _new_chat_id()
    st.session_state.chats          = {first_id: _new_chat()}
    st.session_state.active_chat_id = first_id

# Guard: active_chat_id can go missing if session state is partially
# reset (e.g. hot-reload, Streamlit Cloud rerun). Always ensure it
# points to a valid existing chat.
if "active_chat_id" not in st.session_state or \
        st.session_state.active_chat_id not in st.session_state.get("chats", {}):
    if st.session_state.get("chats"):
        st.session_state.active_chat_id = next(iter(st.session_state.chats))
    else:
        first_id = _new_chat_id()
        st.session_state.chats          = {first_id: _new_chat()}
        st.session_state.active_chat_id = first_id

if "health"    not in st.session_state: st.session_state.health    = None
if "health_ts" not in st.session_state: st.session_state.health_ts = 0
if "confirm_clear_index" not in st.session_state: st.session_state.confirm_clear_index = False

# Convenience accessor — always the currently active chat's message list
def active_messages() -> list:
    cid = st.session_state.active_chat_id
    return st.session_state.chats[cid]["messages"]

def active_name() -> str:
    cid = st.session_state.active_chat_id
    return st.session_state.chats[cid]["name"]


# ── API helpers ───────────────────────────────────────────────────────

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
        yield {"type": "error", "message": "Timed out — API may be cold-starting. Retry in a few seconds."}
    except requests.exceptions.ConnectionError:
        yield {"type": "error", "message": f"Cannot reach {API_BASE}. Check the GCP service is running."}
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


def ingest_confluence(base_url, username, api_token, space_key,
                      max_pages, mock, verify_ssl=True):
    try:
        r = requests.post(f"{API_BASE}/ingest/confluence", timeout=120, json={
            "base_url": base_url, "username": username, "api_token": api_token,
            "space_key": space_key, "max_pages": max_pages, "mock": mock,
            "verify_ssl": verify_ssl,
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


# ── Left sidebar — ingestion + health ────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Assistant")
    st.caption("Internal Knowledge Base")

    # Health
    health = fetch_health()
    if health is None:
        st.markdown('<span class="unhealthy">● API unreachable</span>', unsafe_allow_html=True)
    elif health.get("_cold_start"):
        st.markdown('<span class="warming">● API warming up…</span>', unsafe_allow_html=True)
        st.caption("Cloud Run cold start — refresh in ~10s")
    else:
        doc_count = health.get("documents_indexed", "?")
        cache     = health.get("cache", {})
        st.markdown('<span class="healthy">● API healthy</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card">📄 <b>{doc_count}</b> docs indexed</div>',
                    unsafe_allow_html=True)
        if cache:
            hit_rate = cache.get("hit_rate_pct", cache.get("hit_rate", 0))
            entries  = cache.get("entries", "?")
            st.markdown(
                f'<div class="metric-card">⚡ Cache: <b>{hit_rate}%</b> hit rate · {entries} entries</div>',
                unsafe_allow_html=True)

    if st.button("↻ Refresh status", use_container_width=True):
        fetch_health(force=True)
        st.rerun()

    st.divider()
    top_k = st.slider("Sources to retrieve (top-k)", 2, 10, 5)
    st.divider()

    # Ingestion
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
            cf_url = st.text_input(
                "Base URL  (include context path)",
                value="https://mock.atlassian.net" if use_mock_cf else "",
                placeholder="https://yourorg.atlassian.net/wiki",
                help=(
                    "Paste the root URL of your Confluence instance including the context path.\n\n"
                    "Self-hosted Server → https://yourorg.example.com/confluence\n"
                    "Atlassian Cloud → https://yourorg.atlassian.net/wiki\n\n"
                    "You can also paste a full page URL — the path will be stripped automatically."
                ),
                disabled=use_mock_cf,
            )
            cf_user = st.text_input(
                "Email / username",
                value="demo@acme.com" if use_mock_cf else "",
                placeholder="you@example.com",
                disabled=use_mock_cf,
            )
            cf_token = st.text_input(
                "Password  (Server)  /  API token  (Cloud)",
                value="mock" if use_mock_cf else "",
                type="password",
                help=(
                    "Self-hosted Confluence Server: enter your LOGIN PASSWORD — "
                    "the same one you use to log into the website.\n\n"
                    "Atlassian Cloud (*.atlassian.net): enter an API token from "
                    "id.atlassian.com/manage-profile/security/api-tokens.\n\n"
                    "API tokens do NOT work on self-hosted Server instances."
                ),
                disabled=use_mock_cf,
            )
            cf_verify_ssl = st.checkbox(
                "Verify SSL certificate",
                value=True,
                help=(
                    "Uncheck if your Confluence server uses a self-signed certificate "
                    "and you are getting SSL errors. Keep checked for Atlassian Cloud."
                ),
                disabled=use_mock_cf,
            )
            cf_space = st.text_input(
                "Space key  (or paste any page URL)",
                value="MOCK" if use_mock_cf else "",
                placeholder="ENG",
                help=(
                    "Enter just the space key (e.g. ENG) OR paste a full page URL.\n\n"
                    "Example: https://yourorg.atlassian.net/wiki/spaces/ENG/pages/... "
                    "→ extracts ENG automatically."
                ),
            )
            cf_max = st.number_input("Max pages", 1, 500, 50)
            cf_go  = st.form_submit_button("Ingest Confluence", use_container_width=True)

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
                        verify_ssl=cf_verify_ssl if not use_mock_cf else True,
                    )
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(
                        f"✓ {result.get('pages_fetched', 0)} pages → "
                        f"{result.get('chunks_added', 0)} chunks"
                    )
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
            j_token = st.text_input("API token / password",
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
                    st.success(
                        f"✓ {result.get('issues_fetched', 0)} issues → "
                        f"{result.get('chunks_added', 0)} chunks"
                    )
                    fetch_health(force=True)

    st.divider()

    # ── Index management ──────────────────────────────────────────────
    st.subheader("Index Management")

    if st.button("📋 What's in the index?", use_container_width=True):
        with st.spinner("Fetching index contents…"):
            try:
                r = requests.get(f"{API_BASE}/index/contents", timeout=15)
                r.raise_for_status()
                data = r.json()
                docs = data.get("documents", [])
                st.caption(
                    f"**{data['total_chunks']} chunks** across "
                    f"**{data['unique_documents']} documents**"
                )
                if docs:
                    by_connector: dict = {}
                    for d in docs:
                        c = d.get("connector", "file")
                        by_connector.setdefault(c, []).append(d)
                    for connector, items in sorted(by_connector.items()):
                        icon = {"confluence": "🔵", "jira": "🟠", "file": "📄"}.get(connector, "📄")
                        with st.expander(f"{icon} {connector.capitalize()} ({len(items)} docs)",
                                         expanded=True):
                            for doc in items:
                                st.markdown(
                                    f"**{doc['title']}** — {doc['chunks']} chunk(s)  \n"
                                    f"<span style='font-size:11px;color:grey'>{doc['source']}</span>",
                                    unsafe_allow_html=True,
                                )
                else:
                    st.info("Index is empty. Ingest some documents first.")
            except Exception as e:
                st.error(f"Failed: {e}")

    if st.button("🗑 Clear knowledge base", use_container_width=True, type="secondary"):
        st.session_state.confirm_clear_index = True

    if st.session_state.confirm_clear_index:
        st.warning("This deletes ALL indexed documents. Are you sure?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, clear it", use_container_width=True, type="primary"):
                with st.spinner("Clearing…"):
                    try:
                        r = requests.post(f"{API_BASE}/admin/clear-index", timeout=30)
                        r.raise_for_status()
                        result = r.json()
                        st.success(f"✓ Cleared {result.get('chunks_deleted', 0)} chunks")
                        st.session_state.confirm_clear_index = False
                        fetch_health(force=True)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        with col2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear_index = False
                st.rerun()


# ── Main layout: chat area (left) + chat history panel (right) ────────
# ── Main layout: chat area (left) + chat history panel (right) ────────
chat_col, history_col = st.columns([4, 1], gap="large")

# ── Right column — chat history panel ────────────────────────────────
with history_col:
    st.markdown("#### 💬 Chats")

    if st.button("＋ New chat", use_container_width=True, type="primary"):
        new_id = _new_chat_id()
        st.session_state.chats[new_id] = _new_chat()
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.markdown("---")

    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True,
    )

    for cid, chat in sorted_chats:
        is_active = (cid == st.session_state.active_chat_id)
        msg_count = len(chat["messages"])

        name_col, del_col = st.columns([5, 1], gap="small")
        with name_col:
            label = chat["name"]
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                label,
                key=f"select_{cid}",
                use_container_width=True,
                type=btn_type,
                help=f"{msg_count} message(s) · {chat['created_at']}",
            ):
                if not is_active:
                    st.session_state.active_chat_id = cid
                    st.rerun()

        with del_col:
            if len(st.session_state.chats) > 1:
                if st.button("✕", key=f"del_{cid}", help="Delete chat"):
                    del st.session_state.chats[cid]
                    if st.session_state.active_chat_id == cid:
                        st.session_state.active_chat_id = next(
                            iter(st.session_state.chats)
                        )
                    st.rerun()

        st.caption(f"{msg_count} msg · {chat['created_at']}")

# ── Left column — active chat ─────────────────────────────────────────
with chat_col:
    # Header — just the chat name, no inline rename input
    st.markdown(f"### {active_name()}")

    # Inline rename — collapsible expander so it doesn't occupy permanent space
    with st.expander("✏️ Rename this chat", expanded=False):
        new_name = st.text_input(
            "New name",
            value=active_name(),
            key=f"rename_{st.session_state.active_chat_id}",
            label_visibility="collapsed",
            placeholder="Enter a new name…",
        )
        if st.button("Save name", key=f"save_rename_{st.session_state.active_chat_id}"):
            if new_name.strip():
                st.session_state.chats[st.session_state.active_chat_id]["name"] = new_name.strip()
                st.rerun()

    st.divider()

    # Render message history
    messages = active_messages()
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("from_cache"):
                    st.markdown(
                        '<span class="cache-badge">⚡ From semantic cache</span>',
                        unsafe_allow_html=True,
                    )
                if msg.get("sources"):
                    with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                        for s in msg["sources"]:
                            src     = s.get("source", "unknown")
                            preview = s.get("preview", "")
                            st.markdown(
                                f'<span class="source-chip">📄 {src}</span>',
                                unsafe_allow_html=True,
                            )
                            if preview:
                                st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))

    # Chat input — stays at the bottom naturally since it's last in the column
    # ── Left column — active chat ─────────────────────────────────────────
with chat_col:
    st.markdown(f"### {active_name()}")
    
    # ... rename expander, divider, message history rendering ...
    # STOP HERE — do not put chat_input inside this block

# ── Chat input — OUTSIDE all columns so it pins to page bottom ────────
if question := st.chat_input("Ask a question about your documents…"):
    cid      = st.session_state.active_chat_id
    chat_obj = st.session_state.chats[cid]

    # Auto-name from first question
    if chat_obj["name"] == "New chat" and not chat_obj["messages"]:
        chat_obj["name"] = (question[:35] + "…") if len(question) > 35 else question

    # Append user message
    chat_obj["messages"].append({"role": "user", "content": question})
    
    # Re-render user message in chat_col
    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)

    prior_messages = chat_obj["messages"][:-1]

    with chat_col:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            sources     = []
            from_cache  = False
            error_msg   = None

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
                st.markdown(
                    '<span class="cache-badge">⚡ From semantic cache</span>',
                    unsafe_allow_html=True,
                )
            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=False):
                    for s in sources:
                        src     = s.get("source", "unknown")
                        preview = s.get("preview", "")
                        st.markdown(
                            f'<span class="source-chip">📄 {src}</span>',
                            unsafe_allow_html=True,
                        )
                        if preview:
                            st.caption(preview[:200] + ("…" if len(preview) > 200 else ""))

    if full_answer and not error_msg:
        chat_obj["messages"].append({
            "role":       "assistant",
            "content":    full_answer,
            "sources":    sources,
            "from_cache": from_cache,
        })