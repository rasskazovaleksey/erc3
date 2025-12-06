import logging
import sys
import time

import yaml
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.experts.schemas import ExecutionPlan, PlanStep, ConstraintExpertOutput
from erc.persona import PersonaProvider
from erc.state import AgentState, Plan
from erc.store.tools import TOOLS_DESC


class PlanningExpert(BaseExpert):

    def __init__(self, persona_path, tool_desc: str, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("planning_expert", persona_path)
        self.tools_desc = tool_desc
        self.llm = llm.with_structured_output(ExecutionPlan)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info(f"Planner node started.")

        plan = state.get("plan", None)
        # 1. note main persona is taken from generation
        system_text = self.persona_provider.get_primary_persona()
        # 2. secondary persona is takes from generation
        if not plan:
            user_text = f"""
                Generate a detailed multi-step execution plan to complete the user's task using the tools provided.
                Available Tools:
                {self.tools_desc}
                
                TASK: {state['input_task']}
            """
        else:
            user_text = f"""
                The previous execution plan attempt was invalid.
                Please generate a revised and improved multi-step execution plan to complete the user's task using the tools provided.
                Available Tools:
                {self.tools_desc}
                
                TASK: {state['input_task']}
                
                PREVIOUS PLAN:
                {plan.plan}
                
                REVIEW COMMENTS:
                {plan.review}
            """

        messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        state_copy = state.copy()

        try:
            started = time.time()
            usage_meta_data = UsageMetadataCallbackHandler()
            response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
            logging.info(f"PLANNER RESPONSE: {response}")
            self.callback(usage_meta_data, started)

            execution_candidate: ExecutionPlan = response

            logging.info(f"STRATEGIC PLAN:")
            for i, step in enumerate(execution_candidate.steps):
                logging.info(f"     {i + 1}. {step.tool_name} (Args: {str(step.arguments)[:40]}...)")
                logging.info(f"       Why: {step.reasoning}")
            logging.info("-" * 30)

            plan = Plan(
                plan=execution_candidate,
                is_validated=False,
                validation_attempts=0,
                review=None,
            )
            state_copy["plan"] = plan
            return state_copy
        except Exception as e:
            logging.error(f"PLANNER CRASH: {e}")
            state_copy["plan"] = plan
            return state_copy


if __name__ == "__main__":
    def meta_callback(meta, started):
        pass


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    with open("../../credentials.yml", "r") as f:
        config = yaml.safe_load(f)

    base_url = config["HOST_URL"]

    llm = ChatOpenAI(
        model="oss-20b",
        base_url=base_url,
        api_key="",
        temperature=0.0,
    )
    p = PlanningExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )
    state = AgentState(
        input_task="Count characters in word raspberry",
        plan=None,
    )
    p.node(state)

    invalid_state = AgentState(
        input_task="Count characters is world raspberry",
        plan=Plan(
            plan=ExecutionPlan(steps=[PlanStep(tool_name='report_completion_incorrect',
                                               arguments={'final_message': "The word 'raspberry' has 9 characters."},
                                               reasoning="The task is to count the characters in the word 'raspberry', which is 9. No shopping tools are relevant.",
                                               summary='Completed the character count.')]),
            is_validated=False,
            validation_attempts=0,
            review=ConstraintExpertOutput(
                is_valid=False,
                review_feedback="The plan is invalid and report_completion_incorrect was not found in the available tools."
            ),
        )
    )
    p.node(invalid_state)
