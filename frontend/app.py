# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import json
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Business Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Business Intelligence Agent")
st.caption("Your AI-powered analyst for product performance and customer sentiment.")

# --- Backend API URL ---
AGENT_API_URL = "http://agent_app:8001/ask_agent"

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze your business data today?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# --- Functions ---
def parse_follow_ups(response_text: str) -> list[str]:
    """Extracts bulleted follow-up questions from the agent's response."""
    # This regex is designed to be flexible and capture bulleted questions.
    follow_up_pattern = re.compile(r"^\s*[-*]\s*(.+?\?)\s*$", re.MULTILINE)
    matches = follow_up_pattern.findall(response_text)
    
    # A fallback to find questions under a specific heading
    if not matches:
        try:
            # Find the text after a known follow-up header
            follow_up_section = re.split(r"proactive follow-up|would you like me to|follow-up questions", response_text, flags=re.IGNORECASE)[1]
            lines = follow_up_section.strip().split('\n')
            matches = [line.strip().lstrip('-* ') for line in lines if line.strip().startswith(('-', '*')) and '?' in line]
        except IndexError:
            matches = [] # Section not found
    
    # Final cleanup to remove any markdown formatting from the question itself
    cleaned_matches = [re.sub(r'\[.*?\]\(.*?\)|[*_`]', '', q).strip() for q in matches]
    return cleaned_matches[:4] # Return a max of 4 suggestions

# --- Main App Logic ---

# 1. Display all messages from history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        # For the very last assistant message, add follow-up buttons
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            follow_ups = parse_follow_ups(message["content"])
            if follow_ups:
                st.markdown("**Suggested Follow-ups:**")
                # Use st.columns to prevent buttons from stretching
                cols = st.columns(len(follow_ups))
                for j, question in enumerate(follow_ups):
                    # A unique key is crucial for buttons inside a loop
                    if cols[j].button(question, key=f"follow_up_{i}_{j}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()

# 2. Handle new user input
if prompt := st.chat_input("Ask about sales, reviews, or market trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# 3. If the last message is from the user, get a new response from the agent
if st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking..."):
            try:
                payload = {"query": user_prompt, "session_id": st.session_state.session_id}
                response = requests.post(AGENT_API_URL, json=payload, timeout=300)
                response.raise_for_status() # Raise an exception for bad status codes
                
                data = response.json()
                st.session_state.session_id = data.get("session_id")
                
                # Safely build the markdown content
                summary = data.get("analysis_summary", "No summary provided.")
                recommendations = data.get("actionable_recommendations", [])
                key_insights = data.get("key_insights", [])
                quality_concerns = data.get("data_quality_concerns", [])
                
                # --- FINAL ROBUST RENDERING FIX ---
                # Build each section with aggressive cleaning and guaranteed formatting.
                history_content_md = f"### Analysis Summary\n{summary}\n\n"
                
                if key_insights:
                    history_content_md += "### Key Insights\n"
                    for item in key_insights:
                        # Clean each item before adding it to the list
                        clean_item = str(item).strip().lstrip("-* ")
                        history_content_md += f"- {clean_item}\n"
                    history_content_md += "\n"

                if recommendations:
                    history_content_md += "### Actionable Recommendations\n"
                    for item in recommendations:
                        clean_item = str(item).strip().lstrip("-* ")
                        history_content_md += f"- {clean_item}\n"
                    history_content_md += "\n"

                if quality_concerns:
                    history_content_md += "### Data Quality Concerns\n"
                    for item in quality_concerns:
                        clean_item = str(item).strip().lstrip("-* ")
                        history_content_md += f"- {clean_item}\n"
                    history_content_md += "\n"
                
                # Append the newly generated message and then rerun to display everything
                st.session_state.messages.append({"role": "assistant", "content": history_content_md})
                st.rerun()

            except requests.exceptions.RequestException as e:
                error_text = f"Error connecting to the agent API: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.rerun()
            except Exception as e:
                error_text = f"An unexpected error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.rerun()