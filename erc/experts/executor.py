import logging
import sys
import time

import yaml
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.experts.schemas import ExecutorExpertOutput, ExecutionPlan, PlanStep
from erc.persona import PersonaProvider
from erc.state import AgentState
from erc.store.tools import TOOLS_DESC


class ExecutorExpert(BaseExpert):
    def __init__(self, persona_path, tool_desc: str, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("execution_expert", persona_path)
        self.tools_desc = tool_desc
        self.llm = llm.with_structured_output(ExecutorExpertOutput)
        self.callback = callback

    def node(self, state):
        logging.info("Executor DECIDING...")
        system_text = self.persona_provider.get_primary_persona()
        user_text = f"""
            You are an expert executor that decides the next action to take in order to complete the given task.
            You can write code, use tools, or decide that nothing more is needed.
            Available Tools: {self.tools_desc}
            Task: {state['input_task']}
        """
        started = time.time()
        messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
        usage_meta_data = UsageMetadataCallbackHandler()

        response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})

        logging.info(f"EXECUTOR RESPONSE: {response}")
        self.callback(usage_meta_data, started)

        execution_decision: ExecutorExpertOutput = response

        return state


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
    c = ExecutorExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )

    state = AgentState(
        input_task="Count characters in world raspberry",
        plan=ExecutionPlan(steps=[PlanStep(tool_name='report_completion',
                                           arguments={'final_message': "The word 'raspberry' has 9 characters."},
                                           reasoning="The task is to count the characters in the word 'raspberry', which is 9. No shopping tools are relevant.",
                                           summary='Completed the character count.')]),
        is_validated=False,
        is_valid=False,
        validation_attempts=0,
    )
    c.node(state)
