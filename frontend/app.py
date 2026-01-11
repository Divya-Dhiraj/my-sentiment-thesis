# frontend/app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="HORUS BI Agent", page_icon="🤖", layout="wide")

st.title("🤖 Business Analytics Agent")
st.caption("Your AI-powered analyst for product performance and customer data.")

AGENT_API_URL = "http://agent_app:8001/ask_agent"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze your business data today?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about sales, trends, or returns..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("The agent is thinking..."):
            try:
                payload = {"query": prompt, "session_id": st.session_state.session_id}
                response = requests.post(AGENT_API_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                st.session_state.session_id = data.get("session_id")

                is_complex_response = "key_insights" in data or "table_data" in data or "chart_data" in data
                
                if is_complex_response:
                    summary = data.get("analysis_summary", "")
                    st.markdown(summary)
                    
                    history_content = summary
                    if data.get("key_insights"):
                        insights = "\n".join([f"- {item}" for item in data["key_insights"]])
                        st.markdown("### Key Insights")
                        st.markdown(insights)
                        history_content += f"\n\n### Key Insights\n{insights}"

                    st.session_state.messages.append({"role": "assistant", "content": history_content})

                    table_data = data.get("table_data")
                    if table_data and table_data.get("rows"):
                        st.markdown(f"### {table_data.get('title', 'Data Table')}")
                        df = pd.DataFrame(table_data["rows"], columns=table_data["headers"])
                        st.dataframe(df)

                    chart_data = data.get("chart_data")
                    if chart_data and chart_data.get("data"):
                        st.markdown(f"### {chart_data.get('title', 'Chart')}")
                        df = pd.DataFrame(chart_data['data'])
                        
                        # --- THIS IS THE FIX ---
                        # Check if there's enough data for a chart, otherwise show a table.
                        if len(df) > 1:
                            if not df.empty:
                                df.set_index(df.columns[0], inplace=True)
                            
                            if chart_data.get('type') == 'bar_chart':
                                st.bar_chart(df)
                            else:
                                st.line_chart(df)
                        else:
                            st.write("Displaying data as a table because only one data point is available.")
                            st.dataframe(df)
                else:
                    simple_answer = data.get("analysis_summary", "Sorry, I couldn't find an answer.")
                    st.markdown(simple_answer)
                    st.session_state.messages.append({"role": "assistant", "content": simple_answer})

            except Exception as e:
                error_text = f"An unexpected error occurred: {e}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})