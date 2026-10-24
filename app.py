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
    page_title="Telecom Customer Churn Advisor & Retention Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# Resource Caching
# ==============================================================================
@st.cache_resource(show_spinner="Loading Retention Advisor Agent...")
def get_agent() -> TelecomAdvisorAgent:
    return TelecomAdvisorAgent()


@st.cache_resource(show_spinner="Connecting to Database...")
def get_repository() -> CustomerRepository:
    return CustomerRepository()


@st.cache_resource(show_spinner="Loading ML Churn Predictor...")
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


# ==============================================================================
# State Initialization
# ==============================================================================
repo = get_repository()
predictor = get_predictor()
agent = get_agent()

all_customer_ids = repo.get_all_ids()
if not all_customer_ids:
    st.error("No customer records found in the database. Run 'python run_pipeline.py' to initialize data.")
    st.stop()

# Curated quick presets for peer review & live demonstrations
PRESET_CUSTOMERS = {
    "7590-VHVEG (Moderate Risk: 64.82% - Month-to-Month)": "7590-VHVEG",
    "7216-EWTRS (Critical Risk: 95.45% - High Value Fiber)": "7216-EWTRS",
    "5419-JPRRN (High Risk: 94.46% - Month-to-Month)": "5419-JPRRN",
    "5575-GNVDE (Low Risk: 5.48% - One-Year Contract)": "5575-GNVDE",
    "6928-ONTRW (Stable: 0.22% - Two-Year Loyalty)": "6928-ONTRW",
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
    st.title("Account Lookup")
    st.markdown("Select or search for a customer account to inspect churn metrics and run retention planning.")

    # 1. Preset Quick-select
    st.subheader("Quick Presets")
    preset_choice = st.selectbox(
        "Choose demonstration account:",
        options=["Select a preset..."] + list(PRESET_CUSTOMERS.keys()),
        index=1,
    )
    if preset_choice != "Select a preset...":
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
        "Search customer ID:",
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
        st.subheader("Customer Demographics")
        st.markdown(f"**ID:** `{active_id}`")
        st.markdown(f"**Name:** {customer_record.get('full_name', 'N/A')}")
        st.markdown(f"**Email:** {customer_record.get('email', 'N/A')}")
        st.markdown(f"**Phone:** {customer_record.get('phone', 'N/A')}")
        st.markdown(f"**Account Tenancy:** {customer_record.get('tenure', 0)} months")
        senior_text = "Yes" if customer_record.get("SeniorCitizen") else "No"
        partner_text = "Yes" if customer_record.get("Partner") else "No"
        dependents_text = "Yes" if customer_record.get("Dependents") else "No"
        st.markdown(f"**Senior / Partner / Dependents:** {senior_text} / {partner_text} / {dependents_text}")

    st.markdown("---")

    # 4. Session Controls
    if st.button("Reset Conversation", use_container_width=True):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": (
                    f"Hello! I am ready to advise on customer **{active_id}** "
                    f"({customer_record.get('full_name') if customer_record else ''}). "
                    "You can ask about their churn risk factors, eligible loyalty discounts, "
                    "or contract retention strategies."
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
                "I have loaded their contract parameters and ML churn risk assessment. "
                "How can I assist you with their retention strategy?"
            ),
        }
    ]


# ==============================================================================
# Header Banner
# ==============================================================================
st.title("Telecom Customer Churn Advisor & Retention Console")
st.markdown(
    "AI-powered decision support for frontline retention representatives, grounded in "
    "cost-optimal machine learning thresholds ($t^* = 0.17$) and official retention policies."
)
st.markdown("---")


# ==============================================================================
# Workspace: Two-Column Wide Layout
# ==============================================================================
col1, col2 = st.columns([1.1, 1.2], gap="large")

# ------------------------------------------------------------------------------
# COLUMN 1: Account Diagnostics & Risk Evaluation
# ------------------------------------------------------------------------------
with col1:
    st.header(f"Account Diagnostics: {active_id}")

    if not customer_record:
        st.warning(f"No account details found for customer '{active_id}'.")
    else:
        # 1. Prediction & Risk Evaluation
        pred: Dict[str, Any] = predictor.predict_customer(customer_record)
        is_action_required = pred["is_high_risk"]

        # High-Level Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                label="Churn Probability",
                value=pred["churn_probability_pct"],
                delta=f"{pred['churn_probability'] - 0.17:+.2f} vs t*" if is_action_required else "Normal",
                delta_color="inverse" if is_action_required else "normal",
            )
        with m2:
            st.metric(
                label="Risk Tier",
                value=pred["risk_tier"],
            )
        with m3:
            st.metric(
                label="Decision Cutoff",
                value="t* = 0.17",
                help="Cost-optimal decision threshold minimizing false negatives ($C_FN = $700) vs concessions ($C_FP = $150).",
            )
        with m4:
            st.metric(
                label="Intervention Status",
                value="Action Required" if is_action_required else "Monitoring",
                delta="Flagged" if is_action_required else "Standard",
                delta_color="inverse" if is_action_required else "normal",
            )

        st.markdown("---")

        # 2. Contract & Billing Profile
        st.subheader("Contract & Financial Profile")
        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            st.markdown(f"**Contract Plan:** `{customer_record.get('Contract', 'N/A')}`")
            st.markdown(f"**Monthly Charges:** `${customer_record.get('MonthlyCharges', 0.0):.2f}` / month")
            st.markdown(f"**Total Lifetime Spend:** `${customer_record.get('TotalCharges', 0.0):.2f}`")
        with fin_col2:
            st.markdown(f"**Payment Method:** `{customer_record.get('PaymentMethod', 'N/A')}`")
            paperless = "Enabled" if customer_record.get("PaperlessBilling") else "Disabled"
            st.markdown(f"**Paperless Billing:** `{paperless}`")
            st.markdown(f"**Tenancy Horizon:** `{customer_record.get('tenure', 0)} months`")

        st.markdown("---")

        # 3. Technical Service Inventory
        st.subheader("Active Subscriptions & Service Inventory")

        service_matrix = [
            {"Service Category": "Phone Line", "Subscription": customer_record.get("PhoneService", "No")},
            {"Service Category": "Multiple Lines", "Subscription": customer_record.get("MultipleLines", "No")},
            {"Service Category": "Internet Service", "Subscription": customer_record.get("InternetService", "No")},
            {"Service Category": "Online Security", "Subscription": customer_record.get("OnlineSecurity", "No")},
            {"Service Category": "Online Backup", "Subscription": customer_record.get("OnlineBackup", "No")},
            {"Service Category": "Device Protection", "Subscription": customer_record.get("DeviceProtection", "No")},
            {"Service Category": "Tech Support", "Subscription": customer_record.get("TechSupport", "No")},
            {"Service Category": "Streaming TV", "Subscription": customer_record.get("StreamingTV", "No")},
            {"Service Category": "Streaming Movies", "Subscription": customer_record.get("StreamingMovies", "No")},
        ]
        service_df = pd.DataFrame(service_matrix)
        st.dataframe(service_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------------------
# COLUMN 2: AI Retention Advisor Chat Interface
# ------------------------------------------------------------------------------
with col2:
    st.header("Retention Advisor (ReAct Agent)")
    st.caption(
        "Chat with the advisor to formulate policy-compliant retention strategies, verify discount "
        "eligibility, and check early termination clauses."
    )

    # 1. Render Chat History
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            with st.chat_message(role):
                st.markdown(content)

    # 2. Interactive Input Box
    user_prompt = st.chat_input("Ask about this customer's risk or policy retention offers...")

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
                with st.spinner("Analyzing risk profile and consulting policies..."):
                    try:
                        response_text = agent.chat(
                            message=agent_query,
                            thread_id=f"session_thread_{active_id}",
                            session_state=session_envelope,
                        )
                    except Exception as err:
                        response_text = (
                            f"An error occurred while contacting the advisor: {err}. "
                            "Verify Azure connectivity in .env and retry."
                        )

                    st.markdown(response_text)

        # Append assistant response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
