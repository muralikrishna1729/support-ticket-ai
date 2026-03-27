import streamlit as st
import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title = "SmartTicket AI",
    page_icon  = "🎫",
    layout     = "centered"
)

# ── Session state (stores history across reruns) ──────────────────
if "history" not in st.session_state:
    st.session_state.history = []

st.title(" SmartTicket AI")
st.caption("ML-powered customer support ticket classifier")
st.divider()

ticket = st.text_area(
    "Paste your support ticket here",
    height      = 150,
    placeholder = "e.g. My payment was charged twice and I need a refund"
)

col1, col2 = st.columns([2, 1])
with col1:
    classify_btn = st.button(" Classify Ticket", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

if classify_btn:

    if not ticket.strip():
        st.warning("Please enter a ticket first.")

    else:
        with st.spinner("Classifying..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json = {"ticket": ticket}
                )
                result = response.json()

                
                st.divider()
                st.subheader(" Result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Category",   result["category"])
                with col2:
                    st.metric("Issue Type", result["issue_type"])

                st.info(f" **Auto Response:** {result['auto_response']}")

                
                st.session_state.history.append({
                    "Ticket"        : ticket[:60] + "..." if len(ticket) > 60 else ticket,
                    "Category"      : result["category"],
                    "Issue Type"    : result["issue_type"],
                    "Auto Response" : result["auto_response"]
                })

            except Exception as e:
                st.error(f"Could not connect to API: {str(e)}")


if st.session_state.history:
    st.divider()
    st.subheader(f" Session History ({len(st.session_state.history)} tickets)")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    
    csv = df.to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download History as CSV",
        data      = csv,
        file_name = "ticket_history.csv",
        mime      = "text/csv"
    )
