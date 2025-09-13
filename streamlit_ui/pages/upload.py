import streamlit as st
import requests

st.set_page_config(page_title="Upload File", layout="centered") #  page_icon="📂",
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

uploaded_file = st.file_uploader("Choose a file", type=None)

if uploaded_file is not None:
    st.write("File selected:", uploaded_file.name)

    if st.button("Upload to backend"):
        # Send file to FastAPI endpoint
        url = "http://localhost:8000/upload"  # <-- your FastAPI upload endpoint
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}

        response = requests.post(url, files=files)

        if response.status_code == 200:
            st.success("File uploaded successfully!")
            st.json(response.json())
        else:
            st.error(f"Upload failed: {response.status_code}")
            st.text(response.text)
