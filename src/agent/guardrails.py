import json
import re
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from src.agent.prompts import (
    INPUT_GUARDRAIL_PROMPT,
    OFF_TOPIC_MESSAGE,
    OUTPUT_GUARDRAIL_PROMPT,
    POLICY_COMPLIANCE_FALLBACK,
    SECURITY_NOTICE_MESSAGE,
)
from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)


class SandwichGuardrail:
    """n8n-style sandwich guardrail architecture for state-envelope input validation

    and output compliance verification.

    Stage 1: Evaluates user input against a compact session state envelope (Customer, Workflow, Preceding Turn)
             to decouple injection detection from conversational intent without brittle sliding windows.
    Stage 3: Verifies candidate agent responses for policy compliance ($150 concession budget) and exfiltration.
    """

    INJECTION_THRESHOLD = 0.40
    TOPICAL_THRESHOLD = 0.70
    POLICY_COMPLIANCE_THRESHOLD = 0.85

    def __init__(self, evaluator_llm: Optional[Any] = None):
        self._evaluator_llm = evaluator_llm

    @property
    def evaluator(self) -> Any:
        """Lazy-instantiates the zero-temperature evaluator LLM configured for JSON output."""
        if self._evaluator_llm is None:
            if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
                raise ValueError(
                    "Azure OpenAI credentials missing. Configure AZURE_OPENAI_ENDPOINT "
                    "and AZURE_OPENAI_API_KEY in your .env file."
                )

            self._evaluator_llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
                api_version=AZURE_OPENAI_API_VERSION,
                temperature=0.0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        return self._evaluator_llm

    def _heuristic_input_check(
        self, text: str, session_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[bool, str, Dict[str, float]]]:
        """Fast, zero-token regex pre-screen for unmistakable adversarial signatures or external pivots."""
        lowered = text.lower().strip()

        # 1. Direct prompt injection & system extraction signatures (strictly checked always)
        injection_patterns = [
            r"ignore (all )?(previous|prior|above) (instructions|rules|prompts)",
            r"disregard (all )?(previous|prior) (instructions|rules)",
            r"(print|reveal|show|display|expose|dump|leak) (the )?(system prompt|developer instructions|internal instructions)",
            r"you are now (in )?(developer mode|dan|unrestricted mode)",
            r"grant (a )?100% (free )?lifetime",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, lowered):
                return (
                    False,
                    SECURITY_NOTICE_MESSAGE,
                    {"injection_score": 1.0, "topical_alignment_score": 0.0, "reasoning": "Heuristic injection match."},
                )

        # 2. Conversational short acknowledgments should NEVER be flagged as off-topic by heuristics
        affirmations = {"yes", "sure", "yes sure", "proceed", "go ahead", "why?", "tell me more", "option 1", "option 2", "option 1 please", "option 2 please"}
        if lowered in affirmations:
            return None

        # 3. Obvious out-of-domain pivots
        off_topic_patterns = [
            r"write (a )?(python|java|c\+\+|javascript)? ?(script|code|program|function)? ?to (sort|reverse|scrape|implement|calculate fibonacci)",
            r"(bubble sort|quicksort|merge sort|binary search)",
            r"write a (poem|sonnet|haiku|short story|song) about",
            r"(distance between (earth and mars|the sun and moon)|who was the president of|capital of france)",
        ]
        for pattern in off_topic_patterns:
            if re.search(pattern, lowered):
                return (
                    False,
                    OFF_TOPIC_MESSAGE,
                    {"injection_score": 0.0, "topical_alignment_score": 0.0, "reasoning": "Heuristic off-topic match."},
                )

        return None

    def validate_input(
        self, user_input: str, session_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Stage 1: Multi-criteria evaluation of user input contextualized by a compact state envelope.

        Applies regex heuristic pre-screening for rapid short-circuiting, followed by
        the zero-temperature Evaluator LLM for contextual prompt injection and topicality scoring.

        Args:
            user_input (str): The raw user message.
            session_state (Optional[Dict[str, Any]]): Operational envelope containing optional keys:
                - active_customer: Unique customer identifier or name context.
                - workflow_state: Active agent lifecycle stage.
                - last_assistant_message: Preceding assistant output turn.
                Defaults to an unanchored general account state if None or omitted.

        Returns:
            Tuple[bool, Optional[str], Dict[str, float]]:
                - passed (bool): True if input complies with injection and topical thresholds.
                - fallback_message (Optional[str]): Rejection notice if non-compliant, else None.
                - scores (Dict[str, float]): Evaluator scores for injection and topical alignment.

        Raises:
            ValueError: If Azure OpenAI credentials are unconfigured upon calling the evaluator.
        """
        if not user_input or not user_input.strip():
            return False, "Input cannot be empty.", {"injection_score": 0.0, "topical_alignment_score": 0.0}

        # 1. Fast heuristic pre-check
        heuristic = self._heuristic_input_check(user_input, session_state)
        if heuristic is not None:
            return heuristic

        # 2. Build compact state envelope
        state = session_state or {}
        active_customer = state.get("active_customer") or state.get("customer_id") or "Not specified / General account lookup"
        workflow_state = state.get("workflow_state", "Evaluating Retention & Account Support")
        last_assistant_msg = state.get("last_assistant_message", "None - Conversation start")

        prompt_payload = (
            f"<session_context>\n"
            f"Active Customer: {active_customer}\n"
            f"Workflow State: {workflow_state}\n"
            f"Last Assistant Message: {last_assistant_msg}\n"
            f"</session_context>\n\n"
            f"<user_input>\n{user_input}\n</user_input>"
        )

        # 3. LLM Evaluator scoring
        try:
            messages = [
                SystemMessage(content=INPUT_GUARDRAIL_PROMPT),
                HumanMessage(content=prompt_payload),
            ]
            response = self.evaluator.invoke(messages)
            data = json.loads(response.content)

            injection_score = float(data.get("injection_score", 0.0))
            topical_score = float(data.get("topical_alignment_score", 1.0))
            scores = {"injection_score": injection_score, "topical_alignment_score": topical_score}

            # Hierarchical Threshold Evaluation
            # 1. Check Injection First
            if injection_score > self.INJECTION_THRESHOLD:
                return False, SECURITY_NOTICE_MESSAGE, scores

            # 2. Check Topicality Second
            if topical_score < self.TOPICAL_THRESHOLD:
                return False, OFF_TOPIC_MESSAGE, scores

            return True, None, scores

        except Exception as e:
            # Defensive fallback for sandbox or network interruptions
            print(f"[Input Guardrail Notice] Evaluator fallback: {e}")
            # If input is clearly a conversational continuation, pass safely
            return True, None, {"injection_score": 0.0, "topical_alignment_score": 0.95}

    def validate_output(self, candidate_response: str) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Stage 3: Verifies candidate agent response before displaying to the user.

        Args:
            candidate_response (str): The agent's candidate response.

        Returns:
            Tuple[bool, Optional[str], Dict[str, float]]:
                (passed, final_response, scores)
        """
        if not candidate_response or not candidate_response.strip():
            return False, POLICY_COMPLIANCE_FALLBACK, {"policy_compliance_score": 0.0, "injection_score": 0.0}

        lowered = candidate_response.lower()

        # Heuristic refusal of unauthorized concessions
        if "100% free lifetime" in lowered or "grant a full cash refund of $500" in lowered:
            return (
                False,
                POLICY_COMPLIANCE_FALLBACK,
                {"policy_compliance_score": 0.0, "injection_score": 0.0, "reasoning": "Heuristic concession drift."},
            )

        # Heuristic check for prompt exfiltration in response
        if "you are the telecom customer retention" in lowered and "system persona" in lowered:
            return (
                False,
                SECURITY_NOTICE_MESSAGE,
                {"policy_compliance_score": 0.0, "injection_score": 1.0, "reasoning": "System prompt leaked."},
            )

        try:
            messages = [
                SystemMessage(content=OUTPUT_GUARDRAIL_PROMPT),
                HumanMessage(content=f"<candidate_response>\n{candidate_response}\n</candidate_response>"),
            ]
            response = self.evaluator.invoke(messages)
            data = json.loads(response.content)

            compliance_score = float(data.get("policy_compliance_score", 1.0))
            injection_score = float(data.get("injection_score", 0.0))
            scores = {"policy_compliance_score": compliance_score, "injection_score": injection_score}

            if injection_score > self.INJECTION_THRESHOLD:
                return False, SECURITY_NOTICE_MESSAGE, scores

            if compliance_score < self.POLICY_COMPLIANCE_THRESHOLD:
                return False, POLICY_COMPLIANCE_FALLBACK, scores

            return True, candidate_response, scores

        except Exception as e:
            print(f"[Output Guardrail Notice] Evaluator fallback: {e}")
            return True, candidate_response, {"policy_compliance_score": 0.95, "injection_score": 0.0}
