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

st.title("🎫 SmartTicket AI")
st.caption("ML-powered customer support ticket classifier")
st.divider()

# ── Sidebar Info ──────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    try:
        # Simple health check if your FastAPI has a root / route
        health = requests.get(API_URL, timeout=2)
        if health.status_code == 200:
            st.success("Backend: Connected")
        else:
            st.warning("Backend: Issues detected")
    except:
        st.error("Backend: Disconnected")
    
    st.info("Ensure the FastAPI server is running on port 8000.")

# ── Input Area ────────────────────────────────────────────────────
ticket = st.text_area(
    "Paste your support ticket here",
    height      = 150,
    placeholder = "e.g. My payment was charged twice and I need a refund"
)

col1, col2 = st.columns([2, 1])
with col1:
    classify_btn = st.button("🚀 Classify Ticket", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Main Logic ────────────────────────────────────────────────────
if classify_btn:
    if not ticket.strip():
        st.warning("Please enter a ticket first.")
    else:
        # STEP 1: Submit the ticket
        with st.spinner("Submitting to API..."):
            try:
                response = requests.post(
                    f"{API_URL}/tickets",
                    json={"ticket": ticket},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ticket_id = result.get("id", "N/A")
                    st.info(f"Ticket #{ticket_id} submitted! Status: {result.get('status')}")
                else:
                    # This prevents the JSONDecodeError by showing the raw error instead
                    st.error(f"Backend Error ({response.status_code}): {response.text}")
                    st.stop()

            except requests.exceptions.ConnectionError:
                st.error("Connection Failed: Is your FastAPI server running?")
                st.stop()

        # STEP 2: Poll for the result
        with st.spinner("ML Engine Processing...."):
            progress_bar = st.progress(0)
            found_result = False
            
            for i in range(10): # Try 10 times
                time.sleep(1.5)
                progress_bar.progress((i + 1) * 10)
                
                poll_resp = requests.get(f"{API_URL}/tickets/{ticket_id}")
                if poll_resp.status_code == 200:
                    data = poll_resp.json()
                    
                    if data["status"] == "completed":
                        st.divider()
                        st.subheader("✅ Classification Result")
                        
                        m1, m2 = st.columns(2)
                        m1.metric("Category", data["category"])
                        m2.metric("Issue Type", data["issue_type"])
                        
                        st.success(f"**Auto-Response:** {data['auto_response']}")

                        # Save to history
                        st.session_state.history.append({
                            "Ticket": ticket[:60] + "...",
                            "Category": data["category"],
                            "Issue Type": data["issue_type"],
                            "Time": time.strftime("%H:%M:%S")
                        })
                        found_result = True
                        break
                        
                    elif data["status"] == "failed":
                        st.error(f"Processing failed: {data.get('error', 'Unknown Error')}")
                        found_result = True
                        break
            
            if not found_result:
                st.warning("The system is taking longer than usual. Please check history in a moment.")

# ── History Section ───────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader(f"📊 Session History ({len(st.session_state.history)})")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label     = "⬇️ Download History as CSV",
        data      = csv,
        file_name = "ticket_history.csv",
        mime      = "text/csv"
    )