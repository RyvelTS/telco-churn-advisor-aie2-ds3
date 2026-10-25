"""Telecom Customer Churn Advisor & Retention Console.

Operational Streamlit Dashboard for Customer Service Representatives and Retention Specialists.
"""

from typing import Any, Dict
import pandas as pd
import streamlit as st

from src.agent.advisor_agent import TelecomAdvisorAgent
from src.data.database import CustomerRepository
from src.models.predictor import ChurnPredictor


# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Telecom Churn Advisor | Retention Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI-only styling: keeps the existing business logic and agent pipeline unchanged.
st.markdown(
    """
    <style>
      :root {
        /* Muted sage + blue-grey palette: gentle contrast, readable text. */
        --ui-primary: #6b8f87;
        --ui-primary-deep: #4f746c;
        --ui-primary-soft: #e5efeb;
        --ui-ink: #2d3c3a;
        --ui-muted: #71817d;
        --ui-border: #dfe7e3;
        --ui-surface: #fbfcfb;
        --ui-canvas: #f4f7f5;
        --ui-sidebar: #edf2ef;
      }
      [data-testid="stAppViewContainer"] { background: var(--ui-canvas); color: var(--ui-ink); }
      [data-testid="stHeader"] { background: rgba(244,247,245,0.94); }
      .block-container { padding-top: 1.65rem; padding-bottom: 2.5rem; max-width: 1680px; }
      [data-testid="stSidebar"] { background: var(--ui-sidebar); border-right: 1px solid var(--ui-border); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: var(--ui-ink); }
      h1, h2, h3 { color: var(--ui-ink); letter-spacing: -0.025em; }
      h1 { font-size: clamp(1.6rem, 2.2vw, 2.2rem) !important; line-height: 1.2 !important; }
      h2 { font-size: clamp(1.25rem, 1.65vw, 1.65rem) !important; line-height: 1.25 !important; }
      h3 { font-size: 1.12rem !important; }
      p, li, label { color: var(--ui-ink); }
      [data-testid="stCaptionContainer"] { color: var(--ui-muted); }
      [data-testid="stMetric"] {
        background: var(--ui-surface); border: 1px solid var(--ui-border);
        border-top: 3px solid #c1d7d0; border-radius: 15px;
        padding: 0.9rem 1rem; min-height: 118px;
        box-shadow: 0 3px 10px rgba(50, 75, 68, 0.045);
      }
      [data-testid="stMetricLabel"] { min-height: 1.6rem; }
      [data-testid="stMetricLabel"] p { white-space: normal !important; overflow-wrap: anywhere; color: var(--ui-muted); font-size: 0.82rem !important; }
      [data-testid="stMetricValue"] { white-space: normal !important; overflow-wrap: anywhere; word-break: normal; font-size: clamp(1.25rem, 1.8vw, 1.8rem) !important; line-height: 1.15; color: var(--ui-ink); }
      [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
      [data-testid="stChatMessage"] { border: 1px solid var(--ui-border); border-radius: 14px; padding: 0.9rem 1rem; background: var(--ui-surface); }
      [data-testid="stChatInput"] { border-color: #cbd8d2; background: var(--ui-surface); }
      [data-testid="stChatInput"] textarea { color: var(--ui-ink); }
      div[data-testid="stDataFrame"] { border: 1px solid var(--ui-border); border-radius: 12px; overflow: hidden; }
      div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label { color: #50635e; font-weight: 600; }
      [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #f9fbfa; border-color: #d8e2dd; }
      [data-testid="stTextInput"] input { background: #f9fbfa; border-color: #d8e2dd; }
      button[kind="primary"] { background: var(--ui-primary); border-color: var(--ui-primary); color: white; }
      button[kind="primary"]:hover { background: var(--ui-primary-deep); border-color: var(--ui-primary-deep); }
      hr { border-color: var(--ui-border); }
      .ui-eyebrow { font-size: 0.76rem; font-weight: 700; color: var(--ui-primary-deep); letter-spacing: .09em; text-transform: uppercase; margin-bottom: .35rem; }
      .ui-subtitle { color: var(--ui-muted); margin-top: -0.35rem; margin-bottom: 1.25rem; font-size: .96rem; }
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

    # 1. Render Chat History
    chat_container = st.container(height=430)
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                st.markdown(content)

    # 2. Interactive Input Box
    user_prompt = st.chat_input("Ask about churn risk or retention policies...")

    if user_prompt:
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
                    f"Name: {customer_record.get('full_name', '')}]\n"
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
                "active_customer": f"{active_id}, {customer_record.get('full_name', '')}, {pred.get('risk_tier', '')}",
                "customer_id": active_id,
                "workflow_state": "Evaluating Retention Offer & Account Support",
                "last_assistant_message": last_ai_msg,
            }

            with st.chat_message("assistant"):
                with st.spinner("Analyzing the customer profile and searching policies..."):
                    try:
                        # Initialize the agent only for chat requests; missing Azure
                        # credentials are handled by the existing exception message.
                        agent = get_agent()
                        response_text = agent.chat(
                            message=agent_query,
                            thread_id=f"session_thread_{active_id}",
                            session_state=session_envelope,
                        )
                    except Exception as err:
                        response_text = (
                            "An error occurred while contacting the AI assistant. "
                            "Check the Azure configuration in the .env file and try again. "
                            f"Error type: {type(err).__name__}."
                        )

                    st.markdown(response_text)

        # Append assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
