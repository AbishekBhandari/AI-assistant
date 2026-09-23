import streamlit as st
import requests

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖"
)

st.title("🤖 AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    response = requests.post(
        "http://backend:8000/chat",
        json={"message": prompt}
    )

    if response.status_code == 200:
        data = response.json()
        answer = data["answer"]
    else:
        answer = "Sorry, something went wrong."

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)