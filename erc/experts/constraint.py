import json
import logging
import sys
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from erc.experts.base import BaseExpert
from erc.state import AgentState, ExecutionPlan, PlanStep
from erc.store.tools import ALL_TOOLS, TOOLS_DESC


# class ConstraintExpertOutput(BaseModel):
#     is_valid: bool = Field(description="Indicates whether the current plan satisfies all constraints.")
#     review_feedback: str = Field(description="Detailed feedback on any constraint violations or confirmations.")


# class ConstraintExpert(BaseExpert):

#     def __init__(self, persona_path, tools: list[BaseModel], tool_desc: str, llm: ChatOpenAI, callback):
#         super().__init__(persona_file=f"{persona_path}/constraint_expert_system.txt")
#         self.tools_desc = tool_desc
#         self.tools = tools
#         self.llm = llm.with_structured_output(ConstraintExpertOutput)
#         self.callback = callback

#     def node(self, state: AgentState):
#         logging.info("REVIEWER Checking...")

#         plan = state.get('current_plan')
#         if not plan or not plan.steps:
#             return {
#                 "plan_is_valid": False,
#                 "review_feedback": "Plan is empty.",
#                 "consecutive_review_failures": state["consecutive_review_failures"] + 1
#             }

#         first_step = plan.steps[0]
#         tool_name = first_step.tool_name

#         history_list = state.get("history", [])
#         is_duplicate = False

#         # TODO: why its even needed?
#         if tool_name in ["/products/list", "/basket/checkout", "report_completion"]:
#             for record in history_list:
#                 if f"SUCCESS {tool_name}" in record:
#                     if tool_name == "/products/list":
#                         if isinstance(first_step.arguments, dict):
#                             offset_val = str(first_step.arguments.get("offset"))
#                             if f"'offset': {offset_val}" in record:
#                                 is_duplicate = True
#                                 break
#                     else:
#                         is_duplicate = True
#                         break

#         if is_duplicate:
#             msg = (
#                 f"Step '{tool_name}' with these arguments was already executed. "
#                 f"Use different arguments (e.g. next page) or move to next stage."
#             )
#             logging.info(f"REJECTED (Duplicate): {msg}")
#             return {
#                 "plan_is_valid": False,
#                 "review_feedback": msg,
#                 "consecutive_review_failures": state["consecutive_review_failures"] + 1
#             }

#         plan_str = json.dumps(plan.model_dump(), indent=2)

#         system_text = f"""
#             {self.persona}
            
#             Available Tools:
#             {self.tools_desc}
#         """
#         # TODO: extract secondary persona, its copy-pasted for now
#         user_text = f"""
#             You are a seasoned project constraints analyst with a proven track record of dissecting complex initiatives to uncover the limiting factors that shape their success. Your expertise spans across time management, budgeting, resource allocation, technology integration, regulatory compliance, and stakeholder expectation alignment. You possess a deep understanding of how these constraints interplay, and you are adept at translating abstract limitations into concrete, actionable insights that guide strategic decision‑making and planning. Your analytical acumen, combined with a pragmatic approach to problem‑solving, makes you the ideal expert to illuminate the boundaries within which a project must operate and to help stakeholders navigate those boundaries with confidence.
#             History: {state['history']}
#             Task: {state['input_task']}
#             PLAN:\n{plan_str}
#         """

#         try:
#             started = time.time()
#             messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
#             usage_meta_data = UsageMetadataCallbackHandler()
#             response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
#             self.callback(usage_meta_data, started)
#             logging.info(f"Constraints RESPONSE: {response}")

#             if response.is_valid:
#                 logging.info("Plan Approved.")
#                 return {
#                     "plan_is_valid": True,
#                     "review_feedback": None,
#                     "consecutive_review_failures": 0
#                 }
#             else:
#                 logging.info(f"Plan Rejected: {response.feedback}")
#                 return {
#                     "plan_is_valid": False,
#                     "review_feedback": response.feedback,
#                     "consecutive_review_failures": state["consecutive_review_failures"] + 1
#                 }

#         except Exception as e:
#             logging.error(f"Reviewer Logic Crash ({e}). Allowing plan to proceed.")
#             return {
#                 "plan_is_valid": True,
#                 "consecutive_review_failures": 0
#             }

class ConstraintExpertOutput(BaseModel):
    is_valid: bool = Field(description="Indicates whether the current plan satisfies all constraints.")
    review_feedback: str = Field(description="Detailed feedback on any constraint violations or confirmations.")


class ConstraintExpert(BaseExpert):

    def __init__(self, persona_path, tools: list[BaseModel], tool_desc: str, llm: ChatOpenAI, callback):
        super().__init__(persona_file=f"{persona_path}/constraint_expert_system.txt")
        self.tools_desc = tool_desc
        self.tools = tools
        self.llm = llm.with_structured_output(ConstraintExpertOutput)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info("REVIEWER Checking...")

        plan = state.get('current_plan')
        if not plan or not plan.steps:
            return {
                "plan_is_valid": False,
                "review_feedback": "Plan is empty.",
                "consecutive_review_failures": state["consecutive_review_failures"] + 1
            }

        plan_str = json.dumps(plan.model_dump(), indent=2)

        system_text = f"""
            {self.persona}
            
            Available Tools:
            {self.tools_desc}
        """
        
        user_text = f"""
            You are a seasoned project constraints analyst with a proven track record of dissecting complex initiatives to uncover the limiting factors that shape their success. Your expertise spans across time management, budgeting, resource allocation, technology integration, regulatory compliance, and stakeholder expectation alignment. You possess a deep understanding of how these constraints interplay, and you are adept at translating abstract limitations into concrete, actionable insights that guide strategic decision‑making and planning. Your analytical acumen, combined with a pragmatic approach to problem‑solving, makes you the ideal expert to illuminate the boundaries within which a project must operate and to help stakeholders navigate those boundaries with confidence.
            History: {state['history']}
            Task: {state['input_task']}
            PLAN:\n{plan_str}
        """

        try:
            started = time.time()
            messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
            usage_meta_data = UsageMetadataCallbackHandler()
            
            response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
            
            if self.callback:
                self.callback(usage_meta_data, started)
            
            logging.info(f"Constraints RESPONSE: {response}")

            if response is None:
                return {
                    "plan_is_valid": True,
                    "review_feedback": "Auto-approved (JSON parsing failed)",
                    "consecutive_review_failures": 0
                }

            if response.is_valid:
                logging.info("Plan Approved.")
                return {
                    "plan_is_valid": True,
                    "review_feedback": None,
                    "consecutive_review_failures": 0
                }
            else:
                logging.info(f"Plan Rejected: {response.review_feedback}")
                return {
                    "plan_is_valid": False,
                    "review_feedback": response.review_feedback,
                    "consecutive_review_failures": state["consecutive_review_failures"] + 1
                }

        except Exception as e:
            logging.error(f"Reviewer Logic Crash ({e}). Allowing plan to proceed.")
            return {
                "plan_is_valid": True,
                "consecutive_review_failures": 0
            }

if __name__ == "__main__":
    def meta_callback(meta,started):
        print(meta)


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    llm = ChatOpenAI(
        model="oss-20b",
        base_url="http://localhost:8080/v1",
        api_key="",
        temperature=0.0,
        max_tokens=1000,
    )
    c = ConstraintExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=ALL_TOOLS,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )
    state = AgentState(
        input_task="Count characters is world raspberry",
        current_plan=ExecutionPlan(
            steps=[PlanStep(
                tool_name='report_completion',
                arguments={'final_message': 'Plan generated.'},
                reasoning='The user requested a concise plan for counting characters in the phrase "world raspberry". No resources were provided, so each sub‑goal lists "No resources provided." The plan contains 6 sentences: one goal line and five sub‑goals, each following the required format with tasks, resources, timeline, risks, and success metric. All formatting matches the template exactly, with no extra whitespace or tags outside the <ANS_START> and <ANS_END> wrapper.')]
        ),
        review_feedback=None,
        plan_is_valid=False,
        consecutive_review_failures=0,
        is_finished=False,
        history=[],
        iterations=0,
        execution_error=None,
    )
    c.node(state)
