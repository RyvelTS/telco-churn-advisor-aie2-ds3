from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.agent.guardrails import SandwichGuardrail
from src.agent.prompts import AGENT_SYSTEM_PROMPT
from src.agent.tools import get_customer_churn_profile, search_retention_policies
from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


class TelecomAdvisorAgent:
    """Production Autonomous Agent for Telecom Churn Prevention & Retention Guidance.

    Integrates:
    - n8n-style Sandwich Guardrails (State-Envelope Input & Output Evaluators)
    - ReAct Reasoning Graph (driven by Azure OpenAI gpt-4.1-mini at temperature 0.25)
    - Dynamic Tool Invocation (SQLite CRM Lookup + ML Predictor + Azure Search RAG)
    - Persistent Conversation Memory (MemorySaver checkpointer)
    """

    def __init__(
        self,
        agent_llm: Optional[Any] = None,
        guardrail: Optional[SandwichGuardrail] = None,
    ):
        self.guardrail = guardrail or SandwichGuardrail()

        if agent_llm is None:
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                raise ValueError(
                    "Azure OpenAI credentials missing. Define AZURE_OPENAI_ENDPOINT "
                    "and AZURE_OPENAI_API_KEY in your .env file."
                )

            self.llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=0.25,
            )
        else:
            self.llm = agent_llm

        # Operational tool suite for CRM lookup and policy retrieval
        self.tools = [get_customer_churn_profile, search_retention_policies]

        # Compile ReAct Agent Graph with memory checkpointer
        self.memory = MemorySaver()
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=self.memory,
        )

    def _assemble_state_envelope(
        self,
        thread_id: str,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Constructs a compact operational state envelope (~50 tokens) representing current session context."""
        envelope = dict(session_state) if session_state else {}

        # 1. Retrieve preceding assistant message from memory if not explicitly provided
        if "last_assistant_message" not in envelope:
            config = {"configurable": {"thread_id": thread_id}}
            try:
                state_snapshot = self.graph.get_state(config)
                history = state_snapshot.values.get("messages", []) if state_snapshot else []
                last_ai = next(
                    (
                        m.content
                        for m in reversed(history)
                        if getattr(m, "type", "") == "ai"
                        or getattr(m, "__class__", None).__name__ == "AIMessage"
                    ),
                    None,
                )
                if isinstance(last_ai, list):
                    last_ai = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in last_ai
                    )
                envelope["last_assistant_message"] = last_ai or "None - Conversation start"
            except Exception:
                envelope["last_assistant_message"] = "None - Conversation start"

        # 2. Infer active customer if omitted
        if "active_customer" not in envelope and "customer_id" not in envelope:
            if "thread_" in thread_id:
                cid = thread_id.split("thread_")[-1]
                envelope["active_customer"] = f"Account {cid}"
            else:
                envelope["active_customer"] = "General Customer Inquiry"

        # 3. Default workflow state
        if "workflow_state" not in envelope:
            envelope["workflow_state"] = "Evaluating Customer Retention & Support"

        return envelope

    def chat(
        self,
        message: str,
        thread_id: str = "default",
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Executes the complete sandwich-guarded reasoning lifecycle:

        1. Stage 1: Input Guardrail intercepts malicious or off-topic queries using state-envelope context.
        2. Stage 2: Agent ReAct Engine invokes tools & formulates candidate solution.
        3. Stage 3: Output Guardrail enforces compliance against concessions & prompt leaks.
        4. Returns verified final response.

        Args:
            message (str): User or customer service representative message.
            thread_id (str): Thread ID for conversation memory persistence.
            session_state (Optional[Dict[str, Any]]): Optional state envelope with active customer details.

        Returns:
            str: Verified response or deterministic security fallback string.
        """
        envelope = self._assemble_state_envelope(thread_id, session_state)

        # ======================================================================
        # STAGE 1: State-Envelope Input Guardrail Check
        # ======================================================================
        is_safe, error_msg, _ = self.guardrail.validate_input(message, session_state=envelope)
        if not is_safe:
            return error_msg or ""

        # ======================================================================
        # STAGE 2: ReAct Agent Reasoning & Tool Execution
        # ======================================================================
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=config,
            )
            candidate_response = result["messages"][-1].content
            if isinstance(candidate_response, list):
                # Handle multi-part text blocks if present
                candidate_response = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in candidate_response
                )
        except Exception as e:
            return (
                f"I encountered a temporary service error processing your request: {e}. "
                "Verify system credentials and retry."
            )

        # ======================================================================
        # STAGE 3: Output Guardrail Verification
        # ======================================================================
        is_compliant, verified_response, _ = self.guardrail.validate_output(candidate_response)
        if not is_compliant:
            return verified_response or ""

        return str(verified_response)
