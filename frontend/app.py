# frontend/app.py
import streamlit as st
import requests
import pandas as pd

# Set page configuration for a professional BI look
st.set_page_config(page_title="HORUS BI Agent", page_icon="🤖", layout="wide")

# Custom CSS to improve readability of large tables
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stDataFrame { border: 1px solid #e6e9ef; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 HORUS: Business Intelligence Agent")
st.caption("Analyzing 1,000,000+ records across Apple, Samsung, Google, and more (2023-2026).")

AGENT_API_URL = "http://agent_app:8001/ask_agent"

# Initialize session state for chat history and session tracking
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you analyze the smartphone market trends today?"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ex: 'Why did Samsung Z Fold 7 returns spike in August 2025?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 HORUS is scanning 1M+ records and narratives..."):
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
                    
                    # Key Insights Section
                    history_content = summary
                    if data.get("key_insights"):
                        insights = "\n".join([f"- {item}" for item in data["key_insights"]])
                        st.markdown("### 💡 Key Insights")
                        st.markdown(insights)
                        history_content += f"\n\n### Key Insights\n{insights}"

                    st.session_state.messages.append({"role": "assistant", "content": history_content})

                    # Data Table with Scale Protection
                    table_data = data.get("table_data")
                    if table_data and table_data.get("rows"):
                        st.markdown(f"### 📊 {table_data.get('title', 'Market Data Table')}")
                        df = pd.DataFrame(table_data["rows"], columns=table_data["headers"])
                        
                        # --- BIG DATA PROTECTION ---
                        if len(df) > 50:
                            st.warning(f"Note: Displaying the first 50 of {len(df):,} results for performance.")
                            st.dataframe(df.head(50), use_container_width=True)
                        else:
                            st.dataframe(df, use_container_width=True)

                    # Chart Visualization
                    chart_data = data.get("chart_data")
                    if chart_data and chart_data.get("data"):
                        st.markdown(f"### 📈 {chart_data.get('title', 'Trend Analysis')}")
                        df_chart = pd.DataFrame(chart_data['data'])
                        
                        if len(df_chart) > 1:
                            df_chart.set_index(df_chart.columns[0], inplace=True)
                            if chart_data.get('type') == 'bar_chart':
                                st.bar_chart(df_chart)
                            else:
                                st.line_chart(df_chart)
                        else:
                            st.info("Single data point detected. Displaying as table.")
                            st.dataframe(df_chart)
                else:
                    simple_answer = data.get("analysis_summary", "I couldn't find a specific narrative match.")
                    st.markdown(simple_answer)
                    st.session_state.messages.append({"role": "assistant", "content": simple_answer})

            except Exception as e:
                error_text = f"🚨 Analysis Error: {e}"
                st.error(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})