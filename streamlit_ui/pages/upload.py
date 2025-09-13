import sys
import streamlit as st
import requests
from pathlib import Path

try:
    from core.settings import SETTINGS
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from core.settings import SETTINGS

st.set_page_config(page_title="Upload File", layout="centered")  #  page_icon="📂",
st.title("Upload a File")

st.markdown(
    """
    <style>
    .uploadedFileLimit {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = SETTINGS.UI.API_BASE_URL
ENDPOINT_DOCUMENT_UPLOAD = SETTINGS.UI.ENDPOINT_DOCUMENT_UPLOAD

with st.sidebar:
    st.subheader("Configuration")
    st.text(f"API_BASE_URL = {API_BASE_URL}")
    st.text(f"ENDPOINT_DOCUMENT_UPLOAD = {ENDPOINT_DOCUMENT_UPLOAD}")
    st.caption(
        "Values are loaded from environment (.env). Override by setting env vars."
    )

uploaded_file = st.file_uploader("Choose a file", type=None)

if uploaded_file is not None:
    st.write("File selected:", uploaded_file.name)

    if st.button("Upload to backend"):
        # Send file to FastAPI endpoint
        url = f"{API_BASE_URL}{ENDPOINT_DOCUMENT_UPLOAD}"
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post(url, files=files, timeout=120)
        except requests.RequestException as e:
            st.error(f"Failed to reach API at {url}: {e}")
        else:
            if response.status_code == 200:
                payload = response.json()
                st.success("File uploaded successfully!")
                # Expect ResponseModel with data: DocumentUploadResponse
                data = payload.get("data") if isinstance(payload, dict) else None
                if data:
                    st.write("Document ID:", data.get("document_id"))
                    st.write("Filename:", data.get("filename"))
                    st.write("Size:", data.get("size"))
                    st.write("Checksum:", data.get("checksum"))
                    st.write("Storage Path:", data.get("storage_path"))
                    st.write("Processing Job ID:", data.get("processing_job_id"))
                    with st.expander("Raw Response"):
                        st.json(payload)
                else:
                    st.warning("Unexpected response format; showing raw JSON.")
                    st.json(payload)
            else:
                try:
                    detail = response.json().get("detail")
                except Exception:
                    detail = response.text
                st.error(f"Upload failed: {response.status_code} {detail}")
