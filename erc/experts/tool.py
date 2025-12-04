import logging
import sys
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, ToolCall
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from erc.experts.base import BaseExpert
from erc.state import AgentState, ExecutionPlan, PlanStep
from erc.store.tools import TOOLS_DESC


# TODO: remove me
# see: https://github.com/langchain-ai/langgraph/issues/6397
# def execute_tools(tools: list[BaseTool], message: AIMessage, must_success=True) -> list[ToolMessage]:
#     print("<---- execute_tools MANUAL CALL")
#     tools_by_name = {tool.name: tool
#                      for tool in tools}

#     def _execute_tools(_tool_call: ToolCall) -> ToolMessage:
#         print(_tool_call)
#         _tool = tools_by_name[_tool_call.get("name")]
#         print(_tool)
#         tool_message = _tool.invoke(_tool_call)

#         if must_success:
#             if not isinstance(tool_message.content, str) or not tool_message.content.startswith("SUCCESS"):
#                 raise RuntimeError("toolcall failed")

#         return tool_message

#     return [_execute_tools(tool_call)
#             for tool_call in message.tool_calls]


@tool
def report_task_completion() -> str:
    """
    Reports that the task has been completed successfully.
    """
    # api.report_completion(final_message="Task completed successfully.) <- example just call real client
    print("-------- >>>>>> !!!!Task completed successfully.")
    return "SUCCESS"  # TODO: this is example only, course you hardcoded SUCCESS


# class ExecutorExpers(BaseExpert):

#     def __init__(self, persona_path, tools: list[BaseModel], tool_desc: str, llm: ChatOpenAI, callback):
#         super().__init__(persona_file=f"{persona_path}/tool_expert_system.txt")
#         self.tools_desc = tool_desc
#         self.tools = tools
#         self.llm = llm.bind_tools(self.tools)
#         self.callback = callback

#     def node(self, state: AgentState):
#         logging.info(f"TOOL NODE")
#         plan = state.get('current_plan')
#         step = plan.steps[0]
#         remaining_steps = plan.steps[1:]

#         # 1. note main persona is taken from generation
#         system_text = f"""
#             {self.persona}
#         """
#         # 2. secondary persona is takes from generation
#         # TODO: extract secondary persona, its copy-pasted for now, seems like generated secondary persona is really bad, and prohibits LLM from doing tool calls for get_secret
#         user_text = f"""
#             EXECUTE STEP: {step}
#         """

#         messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]

#         started = time.time()
#         usage_meta_data = UsageMetadataCallbackHandler()
#         response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})
#         self.callback(usage_meta_data, started)

#         logging.info(f"TOOLS RESPONSE: {response}")
#         # TODO: response contains tokens data we need for erc3, wee need to connect it

#         # TODO: wrap tool calling @Viktor, I'm doing example, remove it, shall work good with graph
#         tool_message = execute_tools(self.tools, response, must_success=False)
#         print("<-- tool message", tool_message)
#         # TODO: if content='I’m sorry, but I can’t help with that. retry? or call manually from plan? or mb ToolNode will work better?
#         # tool_node = ToolNode([report_task_completion]) <-- example only
#         # if response.tool_calls:
#         #     print(response.tool_calls)
#         #     tool_node.invoke(response.tool_calls[0])

#         is_completed = response.tool_calls[0]['name'] in ["report_task_completion", "provide_answer"] # TODO: better logic

#         return {
#             "history": state.get("history", []) + tool_message, # TODO: need something better for memory
#             # TODO: @Viktor, please check https://sangeethasaravanan.medium.com/building-tool-calling-agents-with-langgraph-a-complete-guide-ebdcdea8f475
#             "current_plan": ExecutionPlan(steps=remaining_steps),
#             "execution_error": None,
#             "is_finished": is_completed
#         }


class ExecutorExpert(BaseExpert):
    def __init__(self, persona_path, tools: list, tool_desc: str, llm: ChatOpenAI, callback):
        super().__init__(persona_file=f"{persona_path}/tool_expert_system.txt")
        self.llm = llm
        self.callback = callback

    def node(self, state: AgentState):
        logging.info(f"--- EXECUTOR NODE ---")
        
        plan = state.get('current_plan')
        messages = state.get('messages', []) 
        
        if messages and isinstance(messages[-1], ToolMessage):
             logging.info(f"Tool output received: {messages[-1].content}")

        if not plan or not plan.steps:
            logging.info("Executor: No steps left.")
            return {"current_plan": None} 
            
        current_step = plan.steps.pop(0) 
        
        
        system_msg = SystemMessage(content=f"""
        You are an autonomous executor agent.
        Your goal is to execute the next step of the plan.
        
        CRITICAL INSTRUCTION:
        If the plan contains placeholders like '[secret_value]', you MUST check the conversation history (ToolMessages), find the actual value returned by previous tools, and use IT instead of the placeholder.
        """)
        
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

        return {
            "messages": [response], 
            "current_plan": plan 
        }


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
    e = ExecutorExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=[report_task_completion],
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )
    state = AgentState(
        input_task="Count characters is world raspberry",
        current_plan=ExecutionPlan(
            steps=[PlanStep(
                tool_name='report_completion',
                arguments={'final_message': 'Plan generated.'},  # TODO: this gives good debuggable error
                reasoning='The user requested a concise plan for counting characters in the phrase "world raspberry". No resources were provided, so each sub‑goal lists "No resources provided." The plan contains 6 sentences: one goal line and five sub‑goals, each following the required format with tasks, resources, timeline, risks, and success metric. All formatting matches the template exactly, with no extra whitespace or tags outside the <ANS_START> and <ANS_END> wrapper.')]
        ),
        review_feedback="All good.",
        plan_is_valid=True,
        consecutive_review_failures=0,
        is_finished=False,
        history=[],
        iterations=0,
        execution_error=None,
    )
    e.node(state)
