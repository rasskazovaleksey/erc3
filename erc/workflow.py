import logging
import sys

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from erc.experts.constraint import ConstraintExpert
from erc.experts.edges import check_review_status, check_execution_status
from erc.experts.planning import PlanningExpert
from erc.experts.tool import report_task_completion, ExecutorExpert
from erc.state import AgentState
from langchain_core.messages import AIMessage


# def should_continue(state: AgentState):
#     messages = state.get("messages", [])
    
#     if not messages:
#         return "planner"

#     last_message = messages[-1]
    
#     if isinstance(last_message, AIMessage) and last_message.tool_calls:
#         return "tools"

#     return "planner"


# def workflow(
#         planner_node,
#         reviewer_node,
#         executor_node,
# ) -> StateGraph[AgentState]:
#     workflow = StateGraph(AgentState)

#     workflow.add_node("planner", planner_node.node)
#     workflow.add_node("reviewer", reviewer_node.node)
#     #workflow.add_node("executor", executor_node.node)


#     workflow.add_node("tools", executor_node)



#     workflow.set_entry_point("planner")

#     workflow.add_edge("planner", "reviewer")

#     workflow.add_conditional_edges(
#         "executor",
#         should_continue,
#         {"tools": "tools", "end": END}
#     )

#     workflow.add_conditional_edges(
#         "reviewer",
#         check_review_status,
#         {
#             "execute": "executor",
#             "replanning": "planner",
#             END: END
#         }
#     )

#     workflow.add_conditional_edges(
#         "executor",
#         check_execution_status,
#         {
#             "executor": "executor",
#             "planner": "planner",
#             END: END
#         }
#     )

#     return workflow


def check_review_status(state: AgentState):
    """Решает, куда идти после Ревью: выполнять или переделывать."""
    print(f"DEBUG: Check Review Status. Valid? {state.get('plan_is_valid')}")
    
    if state.get("plan_is_valid"):
        return "executor"
    return "planner"

def should_continue(state: AgentState):
    """Решает, куда идти после Executor: вызывать тул или обратно к планировщику."""
    messages = state.get("messages", [])
    if not messages:
        return "planner"

    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"DEBUG: Executor requested tools: {last_message.tool_calls}")
        return "tools"
    
    print("DEBUG: Executor finished step, going back to Planner.")
    return "planner"

def workflow(
    planner_node: PlanningExpert,
    reviewer_node: ConstraintExpert,
    executor_agent: ExecutorExpert, 
    tool_node: ToolNode,           
) -> StateGraph:
    
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node.node)
    workflow.add_node("reviewer", reviewer_node.node)
    workflow.add_node("executor", executor_agent.node) 
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        check_review_status,
        {
            "executor": "executor", 
            "planner": "planner"
        }
    )

    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "tools": "tools",
            "planner": "planner"
        }
    )

    workflow.add_edge("tools", "executor")

    return workflow


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
        persona_path="../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=[report_task_completion],
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )
    c = ConstraintExpert(
        persona_path="../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=[report_task_completion],
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )
    t = ExecutorExpert(
        persona_path="../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tools=[report_task_completion],
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )

    app = workflow(p, c, t).compile()

    logging.info("🚀 Starting Refactored Agent...")
    png_bytes = app.get_graph().draw_mermaid_png()

    # with open('img.png', "wb") as f:
    #     f.write(png_bytes)

    config = RunnableConfig(recursion_limit=50)  # TODO: @Viktor, adjust as needed
    app.invoke({
        "input_task": "Count characters in world raspberry",
        "consecutive_review_failures": 0,
        "history": [],
        "iterations": 0,
        "plan_is_valid": False,
    }, config)
