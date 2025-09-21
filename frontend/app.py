# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import json
import re

st.set_page_config(page_title="Business Intelligence Agent", page_icon="🤖", layout="wide")

st.title("🤖 HORUS: Business Intelligence Agent")
st.caption("Your AI-powered analyst for product performance and customer data.")

AGENT_API_URL = "http://agent_app:8001/ask_agent"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze your business data today?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None

def parse_follow_ups(response_text: str) -> list[str]:
    follow_up_pattern = re.compile(r"^\s*[-*]\s*(.+?\?)\s*$", re.MULTILINE)
    matches = follow_up_pattern.findall(response_text)
    if not matches:
        try:
            follow_up_section = re.split(r"proactive follow-up|would you like me to|follow-up questions|next steps", response_text, flags=re.IGNORECASE)[1]
            lines = follow_up_section.strip().split('\n')
            matches = [line.strip().lstrip('-* ') for line in lines if line.strip().startswith(('-', '*')) and '?' in line]
        except IndexError: matches = []
    cleaned_matches = [re.sub(r'\[.*?\]\(.*?\)|[*_`]', '', q).strip() for q in matches]
    return cleaned_matches[:4]

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            follow_ups = parse_follow_ups(message["content"])
            if follow_ups:
                st.markdown("**Suggested Follow-ups:**")
                cols = st.columns(len(follow_ups))
                for j, question in enumerate(follow_ups):
                    if cols[j].button(question, key=f"follow_up_{i}_{j}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()

if prompt := st.chat_input("Ask about sales, reviews, or market trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

if st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking..."):
            try:
                payload = {"query": user_prompt, "session_id": st.session_state.session_id}
                response = requests.post(AGENT_API_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                st.session_state.session_id = data.get("session_id")
                
                is_complex_response = "key_insights" in data or "actionable_recommendations" in data
                
                if is_complex_response:
                    summary = data.get("analysis_summary", "")
                    recommendations = data.get("actionable_recommendations", [])
                    key_insights = data.get("key_insights", [])
                    quality_concerns = data.get("data_quality_concerns", [])
                    table_data = data.get("table_data")
                    chart_data = data.get("chart_data", {})
                    
                    st.markdown("### Analysis Summary")
                    st.markdown(summary)
                    if key_insights:
                        st.markdown("### Key Insights")
                        for item in key_insights: st.markdown(f"- {str(item).strip().lstrip('-* ')}")
                    if recommendations:
                        st.markdown("### Actionable Recommendations")
                        for item in recommendations: st.markdown(f"- {str(item).strip().lstrip('-* ')}")
                    if quality_concerns:
                        st.markdown("### Data Quality Concerns")
                        for item in quality_concerns: st.markdown(f"- {str(item).strip().lstrip('-* ')}")

                    history_content_md = f"### Analysis Summary\n{summary}\n\n"
                    if key_insights: history_content_md += "### Key Insights\n" + "".join([f"- {str(i).strip().lstrip('-* ')}\n" for i in key_insights]) + "\n"
                    if recommendations: history_content_md += "### Actionable Recommendations\n" + "".join([f"- {str(r).strip().lstrip('-* ')}\n" for r in recommendations]) + "\n"
                    if quality_concerns: history_content_md += "### Data Quality Concerns\n" + "".join([f"- {str(c).strip().lstrip('-* ')}\n" for c in quality_concerns]) + "\n"
                    
                    st.session_state.messages.append({"role": "assistant", "content": history_content_md})

                    if table_data and table_data.get("rows"):
                        st.markdown(f"### {table_data.get('title', 'Summary Table')}")
                        df = pd.DataFrame(table_data["rows"], columns=table_data["headers"])
                        st.dataframe(df)
                    
                    if chart_data and chart_data.get('data'):
                        st.markdown(f"### {chart_data.get('title', 'Chart')}")
                        try:
                            df = pd.DataFrame(chart_data['data'])
                            if 'label' in df.columns and 'label' != df.index.name: df.set_index('label', inplace=True)
                            chart_type = chart_data.get('type', 'bar_chart')
                            if chart_type == 'bar_chart': st.bar_chart(df)
                            elif chart_type == 'line_chart': st.line_chart(df)
                        except Exception as e:
                            st.error(f"Could not display chart. Error: {e}")
                
                else: # Simple, direct answer
                    simple_answer = data.get("analysis_summary", "Sorry, I couldn't find an answer.")
                    st.markdown(simple_answer)
                    st.session_state.messages.append({"role": "assistant", "content": simple_answer})
                
                st.rerun()

            except Exception as e:
                error_text = f"An unexpected error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_text})
                st.rerun()