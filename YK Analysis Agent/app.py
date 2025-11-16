import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from rag_llm import ask_intelligent

# -------- Streamlit Setup --------
st.set_page_config(page_title="YK Intelligent Chat", page_icon="🧠")
st.title("🤖 YK Intelligent RAG Chatbot")

# Secret Key
os.getenv("GROQ_KEY")

# -------- Memory: Full Chat History --------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Context Window (only last 9 messages)
if "memory" not in st.session_state:
    st.session_state.memory = ChatMessageHistory()

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- Chat Input --------
if user_input := st.chat_input("💬 Ask me anything..."):

    # Save and display user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build context
    context_window = st.session_state.messages[-9:]
    context_messages = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in context_window
    ]

    # -------- Streaming Assistant Response --------
    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed_text = ""

        with st.spinner("🤔 Thinking..."):
            try:
                for chunk in ask_intelligent(user_input, context=context_messages):
                    streamed_text += chunk
                    placeholder.markdown(streamed_text)
            except Exception as e:
                streamed_text = f"⚠️ Error: {e}"
                placeholder.markdown(streamed_text)

    # -------- Save Assistant Response --------
    st.session_state.messages.append({"role": "assistant", "content": streamed_text})
    st.session_state.memory.add_message(HumanMessage(user_input))
    st.session_state.memory.add_message(AIMessage(streamed_text))
