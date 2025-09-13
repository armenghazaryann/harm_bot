import streamlit as st
import websocket
import json


st.set_page_config(page_title="LLM Chat", layout="centered") # , page_icon="🤖"
st.title("Chat with HArm Bot")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def stream_response(prompt, placeholder):
    url = "ws://localhost:8000/ws/chat"  # FastAPI WebSocket endpoint
    ws = websocket.WebSocket()
    ws.connect(url)

    payload = {"message": prompt, "history": st.session_state.messages}
    ws.send(json.dumps(payload))

    full_response = ""
    while True:
        try:
            data = ws.recv()
            if data == "[DONE]":
                break

            token = json.loads(data).get("token", data)
            full_response += token
            placeholder.markdown(full_response + "▌")
        except Exception:
            break

    ws.close()
    placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})


# User input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream_response(prompt, placeholder)
