"""Telecom Customer Churn Advisor & Retention Console.

Operational Streamlit Dashboard for Customer Service Representatives and Retention Specialists.
"""

import time
from typing import Any, Dict
import pandas as pd
import streamlit as st

from src.agent.advisor_agent import TelecomAdvisorAgent
from src.data.database import CustomerRepository
from src.models.predictor import ChurnPredictor


# Session Rate Limiting Constants (UX cooldown & token exhaustion defense)
COOLDOWN_SECONDS = 3.0          # Minimum gap between user turns
MAX_PROMPTS_PER_MINUTE = 6     # Maximum turns per minute per session


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Telecom Churn Advisor | Retention Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Client-side theme guarantee: ensure Light theme is selected by default on first visit
st.html(
    """
    <script>
    (function() {
        try {
            const active = window.localStorage.getItem('stActiveTheme');
            if (!active || active === 'system') {
                window.localStorage.setItem('stActiveTheme', 'light');
            }
        } catch (e) {}
    })();
    </script>
    """
)

# Streamlit-native layout & non-destructive enhancements:
# Colors are driven natively by .streamlit/config.toml, avoiding broken widget contrasts.
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.65rem; padding-bottom: 2.5rem; max-width: 1680px; }
      [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
      h1, h2, h3 { letter-spacing: -0.025em; }
      h1 { font-size: clamp(1.6rem, 2.2vw, 2.2rem) !important; line-height: 1.2 !important; }
      h2 { font-size: clamp(1.25rem, 1.65vw, 1.65rem) !important; line-height: 1.25 !important; }
      h3 { font-size: 1.12rem !important; }
      [data-testid="stMetric"] {
        border-top: 3px solid #6b8f87;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 118px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
      }
      [data-testid="stMetricLabel"] { min-height: 1.6rem; }
      [data-testid="stMetricLabel"] p { white-space: normal !important; overflow-wrap: anywhere; font-size: 0.82rem !important; }
      [data-testid="stMetricValue"] { white-space: normal !important; overflow-wrap: anywhere; word-break: normal; font-size: clamp(1.25rem, 1.8vw, 1.8rem) !important; line-height: 1.15; }
      [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
      [data-testid="stChatMessage"] { border-radius: 12px; }
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
      .ui-eyebrow { font-size: 0.76rem; font-weight: 700; color: #4f746c; letter-spacing: .09em; text-transform: uppercase; margin-bottom: .35rem; }
      .ui-subtitle { margin-top: -0.35rem; margin-bottom: 1.25rem; font-size: .96rem; opacity: 0.85; }
      @media (max-width: 900px) {
        .block-container { padding-top: 1rem; }
        [data-testid="stMetric"] { min-height: 105px; padding: .75rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading retention advisor...")
def get_agent() -> TelecomAdvisorAgent:
    try:
        from src.rag.indexer import ensure_index_ready
        ensure_index_ready()
    except Exception:
        pass
    return TelecomAdvisorAgent()


@st.cache_resource(show_spinner="Connecting to database...")
def get_repository() -> CustomerRepository:
    return CustomerRepository()


@st.cache_resource(show_spinner="Loading churn prediction model...")
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


# ==============================================================================
# State Initialization
# ==============================================================================
repo = get_repository()
predictor = get_predictor()
# Load the Azure-backed agent lazily when the user submits a chat prompt.
# This lets the dashboard UI be previewed locally without Azure credentials.

all_customer_ids = repo.get_all_ids()
if not all_customer_ids:
    st.error("Customer data is not available. Run 'python run_pipeline.py' to prepare the database.")
    st.stop()

# Curated quick presets for peer review & live demonstrations
PRESET_CUSTOMERS = {
    "7590-VHVEG · Moderate risk · 64.82% · Month-to-month": "7590-VHVEG",
    "7216-EWTRS · Critical risk · 95.45% · Fiber": "7216-EWTRS",
    "5419-JPRRN · High risk · 94.46% · Month-to-month": "5419-JPRRN",
    "5575-GNVDE · Low risk · 5.48% · One-year contract": "5575-GNVDE",
    "6928-ONTRW · Stable · 0.22% · Two-year contract": "6928-ONTRW",
}

if "selected_customer_id" not in st.session_state:
    st.session_state.selected_customer_id = "7590-VHVEG"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_synced_id" not in st.session_state:
    st.session_state.last_synced_id = None

if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []


# ==============================================================================
# Sidebar: Navigation & Account Lookup
# ==============================================================================
with st.sidebar:
    st.title("Customer Search")
    st.markdown("Select a customer to review churn risk and prepare a retention strategy.")

    # 1. Preset Quick-select
    st.subheader("Quick Presets")
    preset_choice = st.selectbox(
        "Choose a demo account:",
        options=["Choose a demo account..."] + list(PRESET_CUSTOMERS.keys()),
        index=1,
    )
    if preset_choice != "Choose a demo account...":
        chosen_preset_id = PRESET_CUSTOMERS[preset_choice]
        if chosen_preset_id != st.session_state.selected_customer_id:
            st.session_state.selected_customer_id = chosen_preset_id

    # 2. Searchable Account Lookup
    st.subheader("Manual Account Search")
    try:
        current_index = all_customer_ids.index(st.session_state.selected_customer_id)
    except ValueError:
        current_index = 0

    searched_id = st.selectbox(
        "Customer ID:",
        options=all_customer_ids,
        index=current_index,
    )
    if searched_id != st.session_state.selected_customer_id:
        st.session_state.selected_customer_id = searched_id

    active_id = st.session_state.selected_customer_id
    customer_record = repo.get_customer_by_id(active_id)

    st.markdown("---")

    # 3. Customer Demographic Summary
    if customer_record:
        st.subheader("Customer Profile")
        st.markdown(f"**Customer ID:** `{active_id}`")
        st.markdown(f"**Name:** {customer_record.get('full_name', 'Not available')}")
        st.markdown(f"**Email:** {customer_record.get('email', 'Not available')}")
        st.markdown(f"**Phone:** {customer_record.get('phone', 'Not available')}")
        st.markdown(f"**Tenure:** {customer_record.get('tenure', 0)} months")
        senior_text = "Yes" if customer_record.get("SeniorCitizen") else "No"
        partner_text = "Yes" if customer_record.get("Partner") else "No"
        dependents_text = "Yes" if customer_record.get("Dependents") else "No"
        st.markdown(f"**Senior citizen / Partner / Dependents:** {senior_text} / {partner_text} / {dependents_text}")

    st.markdown("---")

    # 4. Session Controls
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    f"Hello! I can help plan a retention strategy for customer **{active_id}** "
                    f"({customer_record.get('full_name') if customer_record else ''}). "
                    "Ask about churn risk, offer eligibility, or contract terms."
                ),
            }
        ]
        st.rerun()


# ==============================================================================
# Synchronize Chat History on Customer Switch
# ==============================================================================
if st.session_state.last_synced_id != active_id:
    st.session_state.last_synced_id = active_id
    cust_name = customer_record.get("full_name", active_id) if customer_record else active_id
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": (
                f"Active account switched to **{active_id}** ({cust_name}). "
                "The contract profile and churn risk assessment are loaded. "
                "What would you like to know about this customer’s retention strategy?"
            ),
        }
    ]


# ==============================================================================
# Header Banner
# ==============================================================================
st.markdown('<div class="ui-eyebrow">Operations Console · Retention Team</div>', unsafe_allow_html=True)
st.title("Telecom Churn Advisor")
st.markdown(
    "Decision-support dashboard to prioritize at-risk customers, review churn predictions, "
    "and prepare retention strategies based on official policies. Model decision threshold: $t^* = 0.17$."
)
st.markdown("---")


# ==============================================================================
# Workspace: Two-Column Wide Layout
# ==============================================================================
col1, col2 = st.columns([1.05, 1.0], gap="large")

# ------------------------------------------------------------------------------
# COLUMN 1: Account Diagnostics & Risk Evaluation
# ------------------------------------------------------------------------------
with col1:
    st.header(f"Risk Summary: {active_id}")

    if not customer_record:
        st.warning(f"Customer details for '{active_id}' were not found.")
    else:
        # 1. Prediction & Risk Evaluation
        pred: Dict[str, Any] = predictor.predict_customer(customer_record)
        is_action_required = pred["is_high_risk"]

        # Spacious 2x2 metric grid prevents long labels and values from being clipped.
        risk_labels = {
            "Low Risk": "Low risk",
            "Moderate Risk": "Moderate risk",
            "High Risk": "High risk",
            "Critical Risk": "Critical risk",
        }
        risk_label = risk_labels.get(pred["risk_tier"], str(pred["risk_tier"]))

        metric_row_1 = st.columns(2, gap="medium")
        with metric_row_1[0]:
            st.metric(
                label="Churn probability",
                value=pred["churn_probability_pct"],
                delta="Above 17% threshold" if is_action_required else "Below 17% threshold",
                delta_color="inverse" if is_action_required else "normal",
            )
        with metric_row_1[1]:
            st.metric(label="Risk tier", value=risk_label)

        metric_row_2 = st.columns(2, gap="medium")
        with metric_row_2[0]:
            st.metric(
                label="Decision threshold",
                value="17% (t*)",
                help="Cost-optimal decision threshold: a false negative is estimated to cost $700, while a retention offer costs $150.",
            )
        with metric_row_2[1]:
            st.metric(
                label="Intervention status",
                value="Action required" if is_action_required else "Monitoring",
                delta="Retention priority" if is_action_required else "Standard",
                delta_color="inverse" if is_action_required else "normal",
            )

        st.markdown("---")

        # 2. Contract & Billing Profile
        st.subheader("Contract & Financial Profile")
        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            st.markdown(f"**Contract plan:** `{customer_record.get('Contract', 'Not available')}`")
            st.markdown(f"**Monthly charges:** `${customer_record.get('MonthlyCharges', 0.0):.2f}` / month")
            st.markdown(f"**Total lifetime spend:** `${customer_record.get('TotalCharges', 0.0):.2f}`")
        with fin_col2:
            st.markdown(f"**Payment method:** `{customer_record.get('PaymentMethod', 'Not available')}`")
            paperless = "Enabled" if customer_record.get("PaperlessBilling") else "Disabled"
            st.markdown(f"**Paperless billing:** `{paperless}`")
            st.markdown(f"**Tenure:** `{customer_record.get('tenure', 0)} months` ")

        st.markdown("---")

        # 3. Technical Service Inventory
        st.subheader("Services Used")

        service_matrix = [
            {"Service type": "Phone service", "Status": customer_record.get("PhoneService", "No")},
            {"Service type": "Multiple lines", "Status": customer_record.get("MultipleLines", "No")},
            {"Service type": "Internet service", "Status": customer_record.get("InternetService", "No")},
            {"Service type": "Online security", "Status": customer_record.get("OnlineSecurity", "No")},
            {"Service type": "Online backup", "Status": customer_record.get("OnlineBackup", "No")},
            {"Service type": "Device protection", "Status": customer_record.get("DeviceProtection", "No")},
            {"Service type": "Tech support", "Status": customer_record.get("TechSupport", "No")},
            {"Service type": "Streaming TV", "Status": customer_record.get("StreamingTV", "No")},
            {"Service type": "Streaming movies", "Status": customer_record.get("StreamingMovies", "No")},
        ]
        service_df = pd.DataFrame(service_matrix)
        status_labels = {
            "Yes": "Yes",
            "No": "No",
            "No internet service": "No internet service",
            "No phone service": "No phone service",
        }
        service_df["Status"] = service_df["Status"].map(lambda value: status_labels.get(str(value), value))
        st.dataframe(service_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# COLUMN 2: AI Retention Advisor Chat Interface
# ------------------------------------------------------------------------------
with col2:
    st.header("AI Retention Advisor")
    st.caption(
        "Ask about customer risk, offer eligibility, or early termination terms. "
        "Policy answers should be verified against source documents."
    )

    def render_policy_citation_cards(sources: list):
        """Formats raw RAG policy chunks into structured citation cards displaying document source and page number."""
        import re

        all_chunks = []
        for src in sources:
            parts = [p.strip() for p in re.split(r"\n*---\n*", str(src)) if p.strip()]
            all_chunks.extend(parts)

        pattern = re.compile(
            r"\[Policy Excerpt \d+ \(Source:\s*(.+?)(?:,\s*Page:\s*-?\d+)?\)\]:\s*([\s\S]+)",
            re.IGNORECASE,
        )

        for i, chunk in enumerate(all_chunks, 1):
            match = pattern.match(chunk)
            if match:
                doc_name, content_text = match.groups()
                badge_html = (
                    f"<div style='margin-bottom: 4px;'>"
                    f"<span style='font-weight: 700; font-size: 0.88rem;'>📄 <code>{doc_name}</code></span>"
                    f"</div>"
                )
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown(f"> {content_text.strip()}")
                if i < len(all_chunks):
                    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px dashed rgba(107, 143, 135, 0.3);'/>", unsafe_allow_html=True)
            else:
                st.markdown(chunk)

    # 1. Render Chat History
    chat_container = st.container(height=430)
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])
            with st.chat_message(role):
                st.markdown(content)
                if role == "assistant" and sources:
                    with st.expander("📄 Retrieved Policy Excerpts (RAG)", expanded=False):
                        render_policy_citation_cards(sources)

    # 2. Interactive Input Box
    user_prompt = st.chat_input("Ask about churn risk or retention policies...")

    if user_prompt:
        now = time.time()

        # 1. Purge timestamps older than 60 seconds
        st.session_state.request_timestamps = [
            t for t in st.session_state.request_timestamps if now - t < 60
        ]

        # 2. Check burst frequency limit (max 6 requests/minute)
        if len(st.session_state.request_timestamps) >= MAX_PROMPTS_PER_MINUTE:
            st.toast("Rate limit reached (max 6 requests/min). Wait a moment.", icon="⏳")
            with chat_container:
                st.warning("Rate limit reached (max 6 requests/min). Wait a moment before sending another message.")
        # 3. Check rapid-fire interval (cooldown gap)
        elif (
            st.session_state.request_timestamps
            and (now - st.session_state.request_timestamps[-1]) < COOLDOWN_SECONDS
        ):
            st.toast(f"Pause {COOLDOWN_SECONDS:.0f} seconds between messages.", icon="⏱️")
        else:
            st.session_state.request_timestamps.append(now)

            # Display user message immediately
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                # 3. Formulate Query with Active Context
                # If user didn't mention customer ID, attach it as context for agent
                if active_id not in user_prompt:
                    agent_query = (
                        f"[Active Customer Context: Account ID '{active_id}', "
                        f"Name: [CUSTOMER_NAME]]\n"
                        f"{user_prompt}"
                    )
                else:
                    agent_query = user_prompt

                # 4. Invoke Agent through State-Envelope Sandwich Guardrails
                last_ai_msg = next(
                    (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "assistant"),
                    "None - Conversation start",
                )
                session_envelope = {
                    "active_customer": f"Account {active_id}, Risk Tier: {pred.get('risk_tier', '')}",
                    "customer_id": active_id,
                    "customer_name": customer_record.get("full_name", "") if customer_record else "",
                    "workflow_state": "Evaluating Retention Offer & Account Support",
                    "last_assistant_message": last_ai_msg,
                }

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing the customer profile and searching policies..."):
                        sources = []
                        try:
                            # Initialize the agent only for chat requests; missing Azure
                            # credentials are handled by the existing exception message.
                            agent = get_agent()
                            result_payload = agent.chat(
                                message=agent_query,
                                thread_id=f"session_thread_{active_id}",
                                session_state=session_envelope,
                                return_details=True,
                            )
                            if isinstance(result_payload, dict):
                                response_text = result_payload.get("response", "")
                                sources = result_payload.get("retrieval_sources", [])
                            else:
                                response_text = str(result_payload)
                        except Exception as err:
                            response_text = (
                                "An error occurred while contacting the AI assistant. "
                                "Check the Azure configuration in the .env file and try again. "
                                f"Error type: {type(err).__name__}."
                            )

                        st.markdown(response_text)
                        if sources:
                            with st.expander("📄 Retrieved Policy Excerpts (RAG)", expanded=False):
                                render_policy_citation_cards(sources)

            # Append assistant response and sources to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response_text,
                "sources": sources,
            })
