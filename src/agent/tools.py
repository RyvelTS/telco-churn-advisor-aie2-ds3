"""LangChain tool definitions for the Telecom Churn Advisor Agent.

Provides typed interfaces to query the SQLite CRM database, execute ML churn predictions,
and search official company retention policies via Azure AI Search.
"""

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.database import CustomerRepository
from src.models.predictor import ChurnPredictor
from src.rag.retriever import AzurePolicyRetriever

# Module-level singletons for performance
_repo: Optional[CustomerRepository] = None
_predictor: Optional[ChurnPredictor] = None
_retriever: Optional[AzurePolicyRetriever] = None


def get_repository() -> CustomerRepository:
    """Returns singleton CustomerRepository instance."""
    global _repo
    if _repo is None:
        _repo = CustomerRepository()
    return _repo


def get_predictor() -> ChurnPredictor:
    """Returns singleton ChurnPredictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor


def get_retriever() -> AzurePolicyRetriever:
    """Returns singleton AzurePolicyRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = AzurePolicyRetriever(top_k=3)
    return _retriever


class CustomerProfileInput(BaseModel):
    """Schema for looking up a customer churn profile."""

    customer_id: str = Field(
        ...,
        description="The unique alphanumeric identifier of the customer (e.g., '7590-VHVEG').",
    )


class PolicySearchInput(BaseModel):
    """Schema for searching company retention policies."""

    query: str = Field(
        ...,
        description=(
            "Keywords or query topic to retrieve relevant retention policies, discount matrices, "
            "competitor price match rules, or early termination fee waivers (e.g., 'fiber upgrade discount', "
            "'early termination fee waiver', 'competitor price match')."
        ),
    )


@tool(args_schema=CustomerProfileInput)
def get_customer_churn_profile(customer_id: str) -> str:
    """Retrieves a customer's CRM profile from the database and runs the ML churn risk model.

    Returns the customer's contract details, monthly charges, current services, tenure,
    predicted churn probability, cost-optimal retention flag (t* = 0.17), and assigned risk tier.

    Use this tool when:
    - You need account details for a specific customer ID.
    - You need to know the customer's churn risk, probability, or intervention eligibility.

    Do NOT use this tool when:
    - You need general policy documents or discount rules (use search_retention_policies instead).
    """
    cleaned_id = customer_id.strip().upper()
    repo = get_repository()
    customer_record = repo.get_customer_by_id(cleaned_id)

    if not customer_record:
        return f"Error: Customer record with ID '{cleaned_id}' not found in the database."

    predictor = get_predictor()
    pred_result = predictor.predict_customer(customer_record)

    # Format comprehensive profile
    services = []
    for svc in [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]:
        val = customer_record.get(svc)
        if val and str(val).lower() not in ["no", "false", "0", "no internet service", "no phone service"]:
            services.append(f"{svc} ({val})")

    active_services_str = ", ".join(services) if services else "None (Basic Phone Only)"

    summary = (
        f"=== Customer Profile & Churn Assessment: {cleaned_id} ===\n"
        f"• Customer Name: [CUSTOMER_NAME]\n"
        f"• Contact: [REDACTED FOR PRIVACY]\n"
        f"• Account Tenure: {customer_record.get('tenure', 0)} months\n"
        f"• Contract Type: {customer_record.get('Contract', 'N/A')}\n"
        f"• Payment Method: {customer_record.get('PaymentMethod', 'N/A')}\n"
        f"• Paperless Billing: {customer_record.get('PaperlessBilling', 'N/A')}\n"
        f"• Monthly Charges: ${customer_record.get('MonthlyCharges', 0.0):.2f}\n"
        f"• Total Charges: ${customer_record.get('TotalCharges', 0.0):.2f}\n"
        f"• Active Services: {active_services_str}\n"
        f"\n--- Machine Learning Churn Analysis (t* = {pred_result['threshold_applied']}) ---\n"
        f"• Churn Probability: {pred_result['churn_probability_pct']} ({pred_result['churn_probability']})\n"
        f"• Retention Action Required: {pred_result['is_high_risk']} (Breaches cost-optimal cutoff {pred_result['threshold_applied']})\n"
        f"• Risk Tier: {pred_result['risk_tier']}\n"
    )
    return summary


@tool(args_schema=PolicySearchInput)
def search_retention_policies(query: str) -> str:
    """Searches official company retention policies, discount structures, and contract terms.

    Queries the Azure AI Search knowledge base and returns relevant excerpts covering
    retention offers, Early Termination Fee (ETF) rules, high-tenure loyalty discounts,
    hardware upgrade terms, and plan comparison guidelines.

    Use this tool when:
    - You need to verify what discounts or offers are officially permitted for a customer's tenure/contract.
    - You need to know early termination fee waiver conditions or competitor matching rules.

    Do NOT use this tool when:
    - Looking up an individual customer's account data (use get_customer_churn_profile instead).
    """
    retriever = get_retriever()
    return retriever.search(query)
