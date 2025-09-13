import sys
from pathlib import Path
import streamlit as st
import requests
from requests.exceptions import RequestException
from requests.exceptions import ConnectionError
from typing import List, Dict, Any
import threading
import queue
import time

try:
    from core.settings import SETTINGS
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.settings import SETTINGS


st.set_page_config(page_title="LLM Chat", layout="centered")  # , page_icon="🤖"
st.title("Chat with HArm Bot")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

API_BASE_URL = SETTINGS.UI.API_BASE_URL
ENDPOINT_QUERY_ANSWER = SETTINGS.UI.ENDPOINT_QUERY_ANSWER

if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = API_BASE_URL


# Helper to create a new conversation via API
def create_conversation():
    """Create a new conversation via the API and return its ID. Tries primary API URL then fallback to localhost if connection fails."""
    primary_url = f"{API_BASE_URL}/api/v1/conversations/"
    fallback_url = "http://localhost:8000/api/v1/conversations/"
    for url in (primary_url, fallback_url):
        try:
            resp = requests.post(url, json={}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            conv = data.get("data", {})
            return conv.get("id")
        except (ConnectionError, RequestException) as e:
            st.error(f"Failed to create conversation at {url}: {e}")
            continue
    st.error("All attempts to create conversation failed.")
    return None


# Initialize conversation
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = create_conversation()


# Helper to fetch recent messages for a conversation
def fetch_messages(conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent messages from the conversation via API and return list of dicts. Tries primary then fallback URL."""
    primary_url = (
        f"{API_BASE_URL}/api/v1/conversations/{conversation_id}/messages?limit={limit}"
    )
    fallback_url = f"http://localhost:8000/api/v1/conversations/{conversation_id}/messages?limit={limit}"
    for url in (primary_url, fallback_url):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json() or {}
            msgs = data.get("data", {}).get("items", [])
            return msgs
        except (ConnectionError, RequestException) as e:
            st.error(f"Failed to fetch messages at {url}: {e}")
            continue
    st.error("All attempts to fetch messages failed.")
    return []


# Helper to append a message to the current conversation
def append_message_to_conversation(conversation_id: str, role: str, content: str):
    """Append a message (user or assistant) to the conversation via API. Tries primary then fallback URL."""
    primary_url = f"{API_BASE_URL}/api/v1/conversations/{conversation_id}/messages"
    fallback_url = (
        f"http://localhost:8000/api/v1/conversations/{conversation_id}/messages"
    )
    payload = {"role": role, "content": content}
    for url in (primary_url, fallback_url):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return True
        except (ConnectionError, RequestException) as e:
            st.error(f"Failed to append message at {url}: {e}")
            continue
    st.error("All attempts to append message failed.")
    return False


# After creating conversation, load existing messages if any
if st.session_state.get("conversation_id") and not st.session_state.messages:
    fetched = fetch_messages(st.session_state["conversation_id"], limit=50)
    # Convert to expected format for display
    for m in fetched:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        st.session_state.messages.append({"role": role, "content": content})

# Sidebar settings
with st.sidebar:
    st.subheader("Configuration")
    st.text(f"API_BASE_URL = {API_BASE_URL}")
    st.text(f"ENDPOINT_QUERY_ANSWER = {ENDPOINT_QUERY_ANSWER}")
    if st.button("New Chat"):
        # Reset messages and create new conversation
        st.session_state.messages = []
        st.session_state.conversation_id = create_conversation()
    st.caption(
        "Values are loaded from environment (.env). Override by setting env vars."
    )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ask_backend(question: str):
    """Call FastAPI answer endpoint and return parsed result or raise on error."""
    url = f"{API_BASE_URL}{ENDPOINT_QUERY_ANSWER}"
    payload = {
        "question": question,
        "context_limit": 5,
        "include_citations": True,
        "temperature": 0.1,
        "max_tokens": 500,
        "detailed": True,
    }
    # Include conversation_id if available
    if st.session_state.get("conversation_id"):
        payload["conversation_id"] = st.session_state["conversation_id"]
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach API at {url}: {e}")

    if resp.status_code != 200:
        # Try extract detail
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        raise RuntimeError(f"API error {resp.status_code}: {detail}")

    data = resp.json() or {}
    if "data" not in data:
        raise RuntimeError("Malformed API response: missing 'data'")
    return data["data"]


# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.get("conversation_id"):
        append_message_to_conversation(
            st.session_state["conversation_id"], "user", prompt
        )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status_placeholder = st.empty()
        # Run backend request in a background thread to allow dynamic status updates
        result_queue = queue.Queue()

        def backend_task():
            try:
                result_queue.put(ask_backend(prompt))
            except Exception as e:
                result_queue.put(e)

        thread = threading.Thread(target=backend_task, daemon=True)
        thread.start()
        # Dynamic spinner while waiting for result
        spinner_chars = ["⏳", "⌛", "⏱️", "🕒"]
        idx = 0
        while thread.is_alive():
            status_placeholder.info(
                f"🔎 Retrieving documents... {spinner_chars[idx % len(spinner_chars)]}"
            )
            time.sleep(0.5)
            idx += 1
        # Retrieve result
        result = result_queue.get()
        if isinstance(result, Exception):
            raise result
        # Step 2: reranking
        status_placeholder.info("🔁 Reranking results...")
        # Step 3: generating answer
        status_placeholder.info("🧩 Generating answer...")
        answer = result.get("answer", "")
        # Step 4: hallucination checking
        status_placeholder.info("🔍 Checking for hallucinations...")
        placeholder.markdown(answer)
        # Append assistant answer to conversation
        if st.session_state.get("conversation_id"):
            append_message_to_conversation(
                st.session_state["conversation_id"], "assistant", answer
            )
        # Render citations if present
        citations = result.get("citations") or []
        if citations:
            with st.expander("Citations"):
                for i, c in enumerate(citations, start=1):
                    doc_name = c.get("document_filename") or "Unknown document"
                    page = c.get("page_number")
                    score = c.get("relevance_score")
                    excerpt = c.get("content_excerpt")
                    st.markdown(
                        f"**{i}. {doc_name}**" + (f", page {page}" if page else "")
                    )
                    st.markdown(f"Score: {score}")
                    if excerpt:
                        st.code(excerpt)
        # Show retrieval diagnostics if present
        if result.get("retrieval_diagnostics"):
            with st.expander("Retrieval Diagnostics"):
                st.json(result["retrieval_diagnostics"])
        # Show verification report if present
        if result.get("verification_report"):
            with st.expander("Verification Report"):
                st.json(result["verification_report"])
        st.session_state.messages.append({"role": "assistant", "content": answer})
        status_placeholder.success("✅ Answer generated")
