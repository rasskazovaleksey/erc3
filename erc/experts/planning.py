import logging
import sys
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from erc.experts.base import BaseExpert
from erc.state import AgentState, ExecutionPlan
from erc.store.tools import ALL_TOOLS, TOOLS_DESC

from utils.utils import get_persona

class PlanningExpert(BaseExpert):

    def __init__(self, persona_path, tools: list[BaseModel], tool_desc: str, llm: ChatOpenAI, callback):
        super().__init__(persona_file=f"{persona_path}/planning_expert_system.txt")
        self.tools_desc = tool_desc
        self.tools = tools
        self.llm = llm.with_structured_output(ExecutionPlan)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info(f"Planner node started.")

        history_list = state.get("history", [])

        finished_actions = []
        #full_history_text = ""

        #for record in history_list:
            #full_history_text += f"- {record}\n" # TODO: WTF is this @Viktor? Memory management shall not be done like this

            # Note:
            # FULL HISTORY: - content='95uZ' name='get_secret' tool_call_id='EakjXvADyXtsiM0qJaLnRDmUDZ7iX6Le'
            # - content='95uZ' name='get_secret' tool_call_id='EakjXvADyXtsiM0qJaLnRDmUDZ7iX6Le'
            # - content='95uZ' name='get_secret' tool_call_id='oxJkRyGWKraTCgbafVt69R1iX3RdHVdu'

            # if record.startswith("SUCCESS"):
            #     clean_rec = record.replace("SUCCESS ", "").split(":")[0]
            #     finished_actions.append(clean_rec)

        finished_str = ", ".join(finished_actions) if finished_actions else "None"

        last_event = history_list[-1] if history_list else ""
        is_error_recovery = "ERROR" in last_event

        # 1. note main persona is taken from generation
        system_text = f"""
            {self.persona}
            
            Available Tools:
            {self.tools_desc}
            
            STATUS:
            - **ALREADY DONE:** [{finished_str}]
            - **Current Goal:** {state['input_task']}
        """
        # 2. secondary persona is takes from generation
        # TODO: extract secondary persona, its copy-pasted for now
        user_text = f"""
            You are a seasoned project‑management strategist with a proven ability to translate ambitious objectives into executable roadmaps. Your background blends Agile, Lean, and traditional Waterfall methodologies, allowing you to tailor the planning process to any context—whether launching a product, implementing a policy, or orchestrating a research initiative. You possess a keen analytical mind for resource allocation, risk assessment, and timeline optimization, and you routinely employ tools like Gantt charts, Kanban boards, and critical‑path analysis to keep projects on track. When presented with a goal, you dissect it into clear milestones, assign responsibilities, anticipate constraints, and devise contingency plans—all while maintaining a concise, actionable narrative that stakeholders can follow and adapt as conditions evolve.
            
            Available Tools:
            {self.tools_desc}
            
            TASK: {state['input_task']}
        """
        print(user_text)

        if state.get("review_feedback") and not state.get("plan_is_valid"):
            user_text += f"""
                PLAN REJECTED (Attempt {state['consecutive_review_failures']}/5): {state['review_feedback']}
                INSTRUCTION: You are likely repeating a step. Advance to the NEXT step.
                """
        elif is_error_recovery:
            user_text += f"""
                LAST STEP FAILED. Please fix the error shown in history.
                """

        messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]

        try:
            started = time.time()
            usage_meta_data = UsageMetadataCallbackHandler()
            response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
            logging.info(f"PLANNER RESPONSE: {response}")
            self.callback(usage_meta_data, started)

            plan: ExecutionPlan = response

            logging.info(f"STRATEGIC PLAN:")
            for i, step in enumerate(plan.steps):
                logging.info(f"     {i + 1}. {step.tool_name} (Args: {str(step.arguments)[:40]}...)")
                logging.info(f"       Why: {step.reasoning}")
            logging.info("-" * 30)

            return {"current_plan": plan, "plan_is_valid": False}
        except Exception as e:
            logging.error(f"PLANNER CRASH: {e}")
            return {"current_plan": None}


if __name__ == "__main__":
    def meta_callback(meta, started):
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
    p = PlanningExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=ALL_TOOLS,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )
    state = AgentState(
        input_task="Count characters is world raspberry",
        current_plan=None,
        review_feedback=None,
        plan_is_valid=False,
        consecutive_review_failures=0,
        is_finished=False,
        history=[],
        iterations=0,
        execution_error=None,
    )
    p.node(state)
