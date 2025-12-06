import json
import logging
import sys
import time

import yaml
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.experts.schemas import ConstraintExpertOutput, PlanStep
from erc.persona import PersonaProvider
from erc.state import AgentState, ExecutionPlan, Plan
from erc.store.tools import TOOLS_DESC


class ConstraintExpert(BaseExpert):

    def __init__(self, persona_path, tool_desc: str, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("constraint_expert", persona_path)
        self.tools_desc = tool_desc
        self.llm = llm.with_structured_output(ConstraintExpertOutput)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info("REVIEWER Checking...")

        plan = state.get("plan", None)
        if not plan or not plan.plan.steps:
            attempts = plan.validation_attempts + 1
            logging.warning("No plan found, auto-rejecting.")
            new_plan = Plan(
                plan=ExecutionPlan(steps=[]),
                is_validated=True,
                validation_attempts=attempts,
                review=ConstraintExpertOutput(
                    is_valid=False,
                    review_feedback="Auto-rejected (no plan found). Please generate a plan."
                )
            )
            return state.copy(
                plan=new_plan,
            )

        plan_str = json.dumps(plan.model_dump(), indent=2)

        system_text = self.persona_provider.get_primary_persona()

        user_text = f"""
            You are an expert reviewer that checks whether the proposed execution plan meets all constraints for the given task.
            Available Tools: {self.tools_desc}
            PLAN:\n{plan_str}
            Task: {state['input_task']}
        """
        try:
            started = time.time()
            messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
            usage_meta_data = UsageMetadataCallbackHandler()

            response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
            logging.info(f"REVIEWER RESPONSE: {response}")
            if self.callback:
                self.callback(usage_meta_data, started)

            state_copy = state.copy()

            plan_copy = plan.model_copy()
            if response is None:
                plan_copy.review = ConstraintExpertOutput(
                    is_valid=False,
                    review_feedback="Auto-rejected (no response from reviewer). Please generate a plan."
                )
                plan_copy.is_validated = True
                plan_copy.validation_attempts += 1

            if response.is_valid:
                plan_copy.review = response
                plan_copy.is_validated = True
                plan_copy.validation_attempts += 1

            else:
                logging.warning(f"Plan Rejected: {response.review_feedback}")
                plan_copy.review = response
                plan_copy.is_validated = True
                plan_copy.validation_attempts += 1

            state_copy["plan"] = plan_copy
            return state_copy

        except Exception as e:
            logging.error(f"Reviewer Logic Crash ({e}). Allowing plan to proceed.")

            state_copy = state.copy()
            plan_copy = plan.model_copy()

            plan_copy.review = ConstraintExpertOutput(
                is_valid=False,
                review_feedback=f"Auto-rejected due to error crash. {e}"
            )
            plan_copy.is_validated = True
            plan_copy.validation_attempts += 1
            state_copy["plan"] = plan_copy
            return state_copy


if __name__ == "__main__":
    def meta_callback(meta, started):
        pass


    with open("../../credentials.yml", "r") as f:
        config = yaml.safe_load(f)

    base_url = config["HOST_URL"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    llm = ChatOpenAI(
        model="oss-20b",
        base_url=base_url,
        api_key="",
        temperature=0.0,
        max_tokens=1000,
    )
    c = ConstraintExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )
    print("""---- GENERATE PLAN ----""")
    state = AgentState(
        input_task="Count characters in world raspberry",
        plan=Plan(
            plan=ExecutionPlan(steps=[PlanStep(tool_name='report_completion',
                                               arguments={'final_message': "The word 'raspberry' has 9 characters."},
                                               reasoning="The task is to count the characters in the word 'raspberry', which is 9. No shopping tools are relevant.",
                                               summary='Completed the character count.')]),
            is_validated=False,
            validation_attempts=0,
            review=None,
        )
    )
    c.node(state)

    print("""---- INVALID PLAN TEST ----""")
    invalid_plan = AgentState(
        input_task="Count characters in world raspberry",
        plan=Plan(
            plan=ExecutionPlan(steps=[PlanStep(tool_name='/products/list', arguments={'offset': 0, 'limit': 1},
                                               reasoning='Retrieve a product list to demonstrate tool usage.',
                                               summary='Fetched product list.'), PlanStep(tool_name='report_completion',
                                                                                          arguments={
                                                                                              'final_message': "The word 'raspberry' has 9 characters."},
                                                                                          reasoning='Provide the final answer to the user.',
                                                                                          summary='Completed the character count.')]),
            is_validated=False,
            validation_attempts=0,
            review=None,
        )
    )
    c.node(invalid_plan)
