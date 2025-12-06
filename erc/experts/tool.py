import logging
import sys
import time

import yaml
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.persona import PersonaProvider
from erc.state import AgentState


class ToolExpert(BaseExpert):
    def __init__(self, persona_path, tools: list, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("feedback_expert", persona_path)
        self.llm = llm.bind_tools(tools)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info(f"ToolExpert Started")

        executor = state.get('executor')
        messages = state.get('messages', [])

        if not executor:
            logging.info("Executor: No steps left.")
            return state

        logging.info(f'executor: {executor}')
        current_step = executor.step

        if messages and isinstance(messages[-1], ToolMessage): 
            logging.info(f"ToolExpert output received: {messages[-1].content}")

        system_msg = SystemMessage(content=self.persona_provider.get_primary_persona())

        context_messages = [system_msg] + messages

        user_text = f"""
        TASK: Execute the next step.
        TOOL: {current_step.tool_name}
        PLANNED ARGUMENTS: {current_step.arguments}
        REASONING: {current_step.reasoning}
        """

        context_messages.append(HumanMessage(content=user_text))

        started = time.time()
        usage_meta_data = UsageMetadataCallbackHandler()

        response = self.llm.invoke(context_messages, config={"callbacks": [usage_meta_data]})

        if self.callback:
            self.callback(usage_meta_data, started)

        logging.info(f"EXECUTOR DECISION: {response.tool_calls}")
        return {"messages": [response]} #TODO better to modify state??

if __name__ == "__main__":
    def meta_callback(meta, started):
        print(meta)


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


    @tool
    def report_task_completion() -> str:
        """
        Reports that the task has been completed successfully.
        """
        # api.report_completion(final_message="Task completed successfully.) <- example just call real client
        print("-------- >>>>>> !!!!Task completed successfully.")
        return "SUCCESS"  # TODO: this is example only, course you hardcoded SUCCESS


    llm = ChatOpenAI(
        model="oss-20b",
        base_url=base_url,
        api_key="",
        temperature=0.0,
        max_tokens=1000,
    )
    e = ToolExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=[report_task_completion],
        callback=meta_callback,
    )
    state = AgentState(
        input_task="Count characters is world raspberry",
    )
    e.node(state)
