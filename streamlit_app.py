# ─────────────────────────────────────────────────────────────────
# DESIGN NOTES (for future edits)
#
# Fonts     : "Space Grotesk"  -> headings / brand / nav (geometric sans)
#             "JetBrains Mono" -> data, metrics, tables, code-like values
#             Both loaded via Google Fonts @import in the injected CSS.
#
# Palette   : dark, technical "dev-tool" direction (Linear / Vercel feel)
#   --bg              #0a0a0f   page background
#   --surface         #131318   card / container background
#   --surface-hover   #1a1a22   hover state for interactive surfaces
#   --border          #24242e   hairline borders
#   --text-primary    #e4e4e7   main text
#   --text-muted      #8a8a99   secondary / caption text
#   --accent          #6366f1   primary action / brand (indigo)
#   --accent-soft     rgba(99,102,241,0.12)  accent tint for backgrounds
#   --success         #22c55e
#   --warning         #f59e0b
#   --danger          #ef4444
#
# Decisions worth knowing:
#   - No functional/business logic was touched: same endpoints, same
#     session_state keys ("history"), same polling loop, same params.
#   - Page identifiers ("Classify Ticket" / "Search Tickets" / "Analytics")
#     lost their emoji prefixes. The `page == "..."` checks were updated
#     to match everywhere they're used -- this is a copy-only change.
#   - st.metric() was replaced with custom HTML stat cards for a more
#     "product dashboard" look. st.dataframe / st.download_button /
#     st.radio remain the same underlying widgets (Streamlit doesn't let
#     us swap those out) but are heavily restyled via CSS.
#   - st.radio is used for nav (unavoidable -- Streamlit has no native
#     nav component). CSS gives it pill-shaped hover states and larger
#     touch targets so it reads like app navigation rather than a form
#     control. True "active route" highlighting isn't fully expressible
#     in pure CSS for st.radio, so this is a close approximation.
#   - Plotly charts use transparent backgrounds + the accent palette so
#     they blend into the surrounding cards instead of showing a white box.
# ─────────────────────────────────────────────────────────────────

import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SmartTicket AI",
    page_icon="◆",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── App styling ───────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0a0a0f;
        --surface: #131318;
        --surface-hover: #1a1a22;
        --border: #24242e;
        --text-primary: #e4e4e7;
        --text-muted: #8a8a99;
        --accent: #6366f1;
        --accent-soft: rgba(99,102,241,0.12);
        --success: #22c55e;
        --success-soft: rgba(34,197,94,0.12);
        --warning: #f59e0b;
        --warning-soft: rgba(245,158,11,0.12);
        --danger: #ef4444;
        --danger-soft: rgba(239,68,68,0.12);
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text-primary);
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.01em;
    }

    p, span, div, label {
        color: var(--text-primary);
    }

    /* ── Brand header ───────────────────────────────────────── */
    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .brand-mark {
        width: 30px;
        height: 30px;
        border-radius: 8px;
        background: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
        color: #fff;
        flex-shrink: 0;
    }
    .brand-name {
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.01em;
    }
    .brand-sub {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-left: 2.35rem;
        margin-top: -0.35rem;
    }

    /* ── Hero / page header card ────────────────────────────── */
    .hero-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.3rem;
    }
    .hero-card h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.5rem;
        color: var(--text-primary);
    }
    .hero-card p {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.92rem;
    }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
        display: block;
    }

    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1.1rem;
    }

    /* ── Stat cards (replace st.metric) ─────────────────────── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.7rem;
        margin-bottom: 0.4rem;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    .stat-card:hover {
        border-color: var(--accent);
        background: var(--surface-hover);
    }
    .stat-label {
        font-size: 0.72rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .stat-value.accent { color: var(--accent); }
    .stat-value.success { color: var(--success); }
    .stat-value.warning { color: var(--warning); }

    /* ── Status pills ────────────────────────────────────────── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.55rem 0.9rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.3rem 0;
    }
    .pill-success { background: var(--success-soft); color: var(--success); }
    .pill-warning { background: var(--warning-soft); color: var(--warning); }
    .pill-danger  { background: var(--danger-soft);  color: var(--danger); }
    .pill-info    { background: var(--accent-soft);  color: var(--accent); }

    .response-box {
        background: var(--accent-soft);
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: var(--text-primary);
        margin-top: 0.5rem;
    }
    .response-box .label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* ── Sidebar ─────────────────────────────────────────────── */
    div[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    div[data-testid="stSidebar"] .stCaption, div[data-testid="stSidebar"] small {
        color: var(--text-muted) !important;
    }

    /* Nav radio styled as app nav */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.25rem;
        display: flex;
        flex-direction: column;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 10px;
        padding: 0.55rem 0.7rem;
        transition: background 0.15s ease;
        width: 100%;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: var(--surface-hover);
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div {
        color: var(--text-primary) !important;
        font-size: 0.9rem;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 0.88rem;
        border: 1px solid var(--border);
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"] {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background: #5254e0;
        border-color: #5254e0;
    }
    .stButton > button[kind="secondary"] {
        background: var(--surface);
        color: var(--text-primary);
    }
    .stButton > button[kind="secondary"]:hover {
        background: var(--surface-hover);
        border-color: var(--accent);
    }

    /* ── Inputs ──────────────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        border-radius: 10px;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    /* ── Alerts (info / warning / success / error) ──────────── */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--surface);
    }

    /* ── Dataframe ───────────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    hr, div[data-testid="stDivider"] {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def check_api_health() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, "API is healthy and ready"
        return False, f"API returned {response.status_code}"
    except requests.RequestException as exc:
        return False, f"Unable to reach API: {exc}"


def stat_card(label: str, value, tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    return f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value{tone_class}">{value}</div>
    </div>
    """


with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
            <div class="brand-mark">ST</div>
            <div class="brand-name">SmartTicket AI</div>
        </div>
        <div class="brand-sub">Ticket intelligence workspace</div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    healthy, message = check_api_health()
    if healthy:
        st.markdown('<span class="pill pill-success">● API Live</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-danger">● API Unreachable</span>', unsafe_allow_html=True)
    st.caption(message)
    st.info("Run FastAPI with:\nuvicorn src.app:app --reload --port 8000")

    st.divider()
    page = st.radio(
        "Navigate",
        ["Classify Ticket", "Search Tickets", "Analytics"],
        label_visibility="collapsed",
    )


if page == "Classify Ticket":
    st.markdown(
        """
        <div class="hero-card">
            <span class="eyebrow">Classify</span>
            <h1>Ticket Classification</h1>
            <p>Turn incoming support requests into structured, actionable insights with confidence scoring.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    samples = {
        "-- Type your own --": "",
        "Billing Issue": "My payment was charged twice this month and I need a refund immediately.",
        "Technical Problem": "The application keeps crashing every time I try to upload a file.",
        "Return Request": "I received a damaged product and want to return it for a replacement.",
        "IT Support": "I cannot login to the company VPN since yesterday morning.",
        "Service Outage": "Your website has been down for 2 hours. We are losing sales.",
    }

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        selected = st.selectbox("Try a sample", list(samples.keys()), label_visibility="collapsed")
        ticket = st.text_area(
            "Paste your support ticket",
            value=samples[selected],
            height=180,
            placeholder="e.g. My payment was charged twice...",
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            classify_btn = st.button("Classify Ticket", type="primary", width="stretch")
        with col2:
            if st.button("Clear History", width="stretch"):
                st.session_state.history = []
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    if classify_btn:
        if not ticket.strip():
            st.warning("Please enter a ticket before submitting.")
        else:
            with st.spinner("Submitting ticket for classification..."):
                try:
                    response = requests.post(
                        f"{API_URL}/tickets",
                        json={"ticket": ticket},
                        timeout=10,
                    )
                    if response.status_code != 200:
                        st.error("Classification request failed.")
                        st.stop()

                    result = response.json()
                    ticket_id = result["id"]

                    for _ in range(10):
                        time.sleep(1)
                        poll = requests.get(f"{API_URL}/tickets/{ticket_id}", timeout=5).json()

                        if poll["status"] == "completed":
                            st.divider()
                            st.markdown('<div class="section-card">', unsafe_allow_html=True)
                            st.markdown('<span class="eyebrow">Result</span>', unsafe_allow_html=True)
                            st.subheader("Classification Result")

                            st.markdown(
                                f"""
                                <div class="stat-grid">
                                    {stat_card("Category", poll["category"])}
                                    {stat_card("Issue Type", poll["issue_type"])}
                                    {stat_card("Confidence", f"{poll['confidence']:.2f}", "accent")}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            if poll["needs_review"]:
                                st.markdown(
                                    '<span class="pill pill-warning">▲ Low confidence — flagged for human review</span>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    '<span class="pill pill-success">✓ High confidence prediction</span>',
                                    unsafe_allow_html=True,
                                )

                            st.markdown(
                                f"""
                                <div class="response-box">
                                    <span class="label">Auto Response</span>
                                    {poll['auto_response']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.session_state.history.append(
                                {
                                    "ID": poll["id"],
                                    "Ticket": ticket[:60] + "..." if len(ticket) > 60 else ticket,
                                    "Category": poll["category"],
                                    "Issue Type": poll["issue_type"],
                                    "Confidence": round(poll["confidence"], 2),
                                    "Needs Review": poll["needs_review"],
                                }
                            )
                            st.markdown("</div>", unsafe_allow_html=True)
                            break

                        if poll["status"] == "failed":
                            st.error("Processing failed.")
                            break
                    else:
                        st.warning("Still processing — try again in a moment.")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API. Make sure FastAPI is running on port 8000.")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    if st.session_state.history:
        st.divider()
        st.markdown('<span class="eyebrow">History</span>', unsafe_allow_html=True)
        st.subheader(f"Session History ({len(st.session_state.history)} tickets)")
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False),
            file_name="ticket_history.csv",
            mime="text/csv",
        )

elif page == "Search Tickets":
    st.markdown(
        """
        <div class="hero-card">
            <span class="eyebrow">Search</span>
            <h1>Search Tickets</h1>
            <p>Search historical tickets by keyword, category, or status.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        keyword = st.text_input("Keyword")
    with col2:
        category = st.selectbox(
            "Category",
            [
                "All",
                "Billing and Payments",
                "Technical Support",
                "IT Support",
                "Customer Service",
                "Product Support",
                "Returns and Exchanges",
                "Service Outages and Maintenance",
                "Sales and Pre-Sales",
                "Human Resources",
                "General Inquiry",
            ],
        )
    with col3:
        status = st.selectbox("Status", ["All", "completed", "pending", "processing", "failed"])

    if st.button("Search", type="primary"):
        params = {}
        if keyword:
            params["keyword"] = keyword
        if category != "All":
            params["category"] = category
        if status != "All":
            params["status"] = status

        try:
            response = requests.get(f"{API_URL}/tickets/search", params=params, timeout=5)
            data = response.json()

            if not data:
                st.info("No tickets found for the selected filters.")
            else:
                st.markdown(
                    f'<span class="pill pill-success">✓ Found {len(data)} tickets</span>',
                    unsafe_allow_html=True,
                )
                df = pd.DataFrame(data)
                st.dataframe(
                    df[["id", "ticket_text", "category", "issue_type", "confidence", "needs_review", "status", "created_at"]],
                    width="stretch",
                    hide_index=True,
                )
        except Exception as exc:
            st.error(f"Error: {exc}")

elif page == "Analytics":
    st.markdown(
        """
        <div class="hero-card">
            <span class="eyebrow">Analytics</span>
            <h1>Analytics Dashboard</h1>
            <p>Monitor volume, outcomes, and ticket quality at a glance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        data = requests.get(f"{API_URL}/analytics/summary", timeout=5).json()

        st.markdown(
            f"""
            <div class="stat-grid">
                {stat_card("Total Tickets", data["total"])}
                {stat_card("Completed", data["completed"], "success")}
                {stat_card("Pending", data["pending"], "warning")}
                {stat_card("Needs Review", data["needs_review"], "warning")}
                {stat_card("Avg Confidence", f"{data['avg_confidence']:.2f}", "accent")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        col1, col2 = st.columns(2)

        chart_font = dict(family="JetBrains Mono, monospace", color="#8a8a99", size=12)

        with col1:
            if data["category_counts"]:
                df_cat = pd.DataFrame(list(data["category_counts"].items()), columns=["Category", "Count"]).sort_values("Count", ascending=True)
                fig = px.bar(
                    df_cat,
                    x="Count",
                    y="Category",
                    orientation="h",
                    title="Tickets by Category",
                    color="Count",
                    color_continuous_scale=["#24242e", "#6366f1"],
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=chart_font,
                    title_font=dict(family="Space Grotesk, sans-serif", color="#e4e4e7", size=15),
                    xaxis=dict(gridcolor="#24242e", zerolinecolor="#24242e"),
                    yaxis=dict(gridcolor="#24242e"),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.markdown(
                    '<div class="section-card"><span class="pill pill-info">No data yet — classify some tickets first.</span></div>',
                    unsafe_allow_html=True,
                )

        with col2:
            if data["issue_type_counts"]:
                df_type = pd.DataFrame(list(data["issue_type_counts"].items()), columns=["Issue Type", "Count"])
                fig2 = px.pie(
                    df_type,
                    names="Issue Type",
                    values="Count",
                    title="Tickets by Issue Type",
                    color_discrete_sequence=["#6366f1", "#8b8ff0", "#4c4fc7", "#a5a8f5", "#34348f", "#c2c4f9"],
                )
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=chart_font,
                    title_font=dict(family="Space Grotesk, sans-serif", color="#e4e4e7", size=15),
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(font=dict(color="#8a8a99")),
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                st.markdown(
                    '<div class="section-card"><span class="pill pill-info">No data yet — classify some tickets first.</span></div>',
                    unsafe_allow_html=True,
                )

    except Exception as exc:
        st.error(f"Could not load analytics: {exc}")
        st.info("Make sure FastAPI is running on port 8000")