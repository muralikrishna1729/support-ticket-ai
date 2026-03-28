import streamlit as st
import requests
import pandas as pd
import time

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
    classify_btn = st.button(" Classify Ticket", type="primary", width='stretch')
with col2:
    if st.button("🗑️ Clear History", width='stretch'):
        st.session_state.history = []
        st.rerun()

if classify_btn:

    if not ticket.strip():
        st.warning("Please enter a ticket first.")

    else:
        with st.spinner("Classifying..."):
            response = requests.post(
                f"{API_URL}/tickets",
                json={"ticket": ticket}
            )
            result = response.json()
            ticket_id = result["id"]
            st.info(f"Ticket #{ticket_id} submitted! Status: {result['status']}")

        with st.spinner("Processing...."):
            for _ in range(10):
                time.sleep(1)
                poll = requests.get(f"{API_URL}/tickets/{ticket_id}")
                data = poll.json()
                if data["status"] == "completed":
                    st.divider()
                    st.subheader(" Result")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Category",   data["category"])
                    with col2:
                        st.metric("Issue Type", data["issue_type"])
                    st.info(f" {data['auto_response']}")

                    # Save history
                    st.session_state.history.append({
                        "Ticket"     : ticket[:60],
                        "Category"   : data["category"],
                        "Issue Type" : data["issue_type"],
                        "Status"     : data["status"]
                    })
                    break

                elif data["status"] == "failed":
                    st.error("Processing failed.")
                    break
            else:
                st.warning("Still processing — check back in a moment.")


if st.session_state.history:
    st.divider()
    st.subheader(f" Session History ({len(st.session_state.history)} tickets)")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, width='stretch', hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download History as CSV",
        data      = csv,
        file_name = "ticket_history.csv",
        mime      = "text/csv"
    )
