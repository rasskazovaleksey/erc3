import logging
import sys
import time

import yaml
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.experts.executor import ExecutorExpert
from erc.experts.schemas import ExecutorExpertOutput, ExecutionPlan, PlanStep
from erc.persona import PersonaProvider
from erc.state import AgentState, ExecutionTool, Plan
from erc.store.tools import TOOLS_DESC


class ReflectionExpert(BaseExpert): 
    def __init__(self, 
                 # persona_path, 
                 # tool_desc: str, 
                 # llm: ChatOpenAI, 
                 # callback
                 ):
        #self.persona_provider = PersonaProvider("", persona_path) #TODO which expert expert????
        #self.tools_desc = tool_desc
        #self.llm = llm.with_structured_output(ExecutorExpertOutput) #TODO do we need LLM here?
        #self.callback = callback
        pass

    def node(self, state):
        logging.info("ReflectionExpert")


        state_copy = state.copy()
        if state_copy['executor'].status == 'SUCCESS': #TODO hardcoded?
            logging.info(f"state: {state}")
            state_copy['step_pointer'] += 1

        return state_copy

