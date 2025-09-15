# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import json

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

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # The content is now rendered directly as it's stored as markdown
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Chat Input ---
if prompt := st.chat_input("Ask about sales, reviews, or market trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        full_response_md = ""
        try:
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }

            with st.spinner("The agent is thinking..."):
                response = requests.post(AGENT_API_URL, json=payload, timeout=300)

            if response.status_code == 200:
                data = response.json()
                st.session_state.session_id = data.get("session_id")

                if "analysis_summary" in data:
                    summary = data.get("analysis_summary", "No summary provided.")
                    recommendations = data.get("actionable_recommendations", [])
                    chart_data = data.get("chart_data", {})
                    
                    full_response_md += f"### Analysis Summary\n{summary}\n\n"
                    
                    if recommendations:
                        full_response_md += "### Actionable Recommendations\n"
                        for rec in recommendations:
                            full_response_md += f"- {rec}\n"
                        full_response_md += "\n"
                    
                    st.markdown(full_response_md)

                    if chart_data and chart_data.get('data') and len(chart_data['data']) > 0:
                        st.write(f"### {chart_data.get('title', 'Chart')}")
                        df = pd.DataFrame(chart_data['data'])
                        
                        if 'label' in df.columns:
                            df.set_index('label', inplace=True)
                        
                        chart_type = chart_data.get('type', 'bar_chart')
                        if chart_type == 'bar_chart':
                            st.bar_chart(df)
                        elif chart_type == 'line_chart':
                            st.line_chart(df)
                else:
                    full_response_md = "Sorry, I received an unexpected response from the agent."
                    st.markdown(full_response_md)
            else:
                full_response_md = f"Error: Received status code {response.status_code}\n\n{response.text}"
                st.markdown(full_response_md)

        except Exception as e:
            full_response_md = f"An error occurred: {e}"
            st.markdown(full_response_md)
        
        # Add the complete response to history for proper reruns
        st.session_state.messages.append({"role": "assistant", "content": full_response_md})