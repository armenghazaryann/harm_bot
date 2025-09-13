import sys
from pathlib import Path
import streamlit as st
import requests

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

# Sidebar settings
with st.sidebar:
    st.subheader("Configuration")
    st.text(f"API_BASE_URL = {API_BASE_URL}")
    st.text(f"ENDPOINT_QUERY_ANSWER = {ENDPOINT_QUERY_ANSWER}")
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
    }
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
    # Expect ResponseModel: { status, message, data: { AnswerResponse... } }
    if "data" not in data:
        raise RuntimeError("Malformed API response: missing 'data'")
    return data["data"]


# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                result = ask_backend(prompt)
                answer = result.get("answer", "")
                placeholder.markdown(answer)
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
                                f"**{i}. {doc_name}**"
                                + (f", page {page}" if page else "")
                            )
                            st.markdown(f"Score: {score}")
                            if excerpt:
                                st.code(excerpt)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                placeholder.error(str(e))
