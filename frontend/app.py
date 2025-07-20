# frontend/app.py
import streamlit as st
import requests
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Business Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Business Intelligence Agent")
st.caption("Your AI-powered analyst for product performance and customer sentiment.")

# --- Backend API URL ---
# This URL uses the Docker service name 'agent_app' which is resolvable inside the Docker network
AGENT_API_URL = "http://agent_app:8001/ask_agent"

# --- Session State Initialization ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze your business data today?"}]
# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask about sales, reviews, or market trends..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Prepare the request payload
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }

            # Send request to the backend
            with st.spinner("The agent is thinking..."):
                response = requests.post(AGENT_API_URL, json=payload, timeout=300)

            if response.status_code == 200:
                data = response.json()
                full_response = data.get("response", "Sorry, I couldn't get a response.")
                # Update the session_id for conversation continuity
                st.session_state.session_id = data.get("session_id")
            else:
                full_response = f"Error: Received status code {response.status_code}\n\n{response.text}"
            
            message_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Could not connect to the agent's backend. Please ensure it's running. \n\n**Error:** {e}"
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})