import logging
import sys
from typing import Annotated

import yaml
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import render_text_description, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode

from erc.experts.constraint import ConstraintExpert
from erc.experts.executor import ExecutorExpert
from erc.experts.planning import PlanningExpert
from erc.experts.reflection import ReflectionExpert
from erc.experts.tool import ToolExpert
from erc.state import AgentState




@tool
def report_task_completion(state: Annotated[dict, InjectedState]) -> str:  #validation error with AgentState, fixed with dict
    """
    Reports that the task has been completed successfully.
    """
    try:
        #state_copy = state.copy() 
        # api.report_completion(final_message="Task completed successfully.) <- example just call real client
        print("-------- >>>>>> !!!!Task completed successfully.")
        state['executor'].status = 'SUCCESS'
        return "SUCCESS"  # TODO: this is example only, course you hardcoded SUCCESS

    except Exception as e:
        state['executor'].status = e
        return 'ERROR'

tools = [report_task_completion]
tools_desc_str = render_text_description(tools)
PATH_PREFFIX = ''#'../../'

def plan_review_loop(state: AgentState):
    logging.info("EDGE: planning review_loop")

    plan = state.get("plan", None)
    if not plan:
        raise Exception("PLANNER EXECUTOR NODE FAILED")

    if plan.validation_attempts >= 5:
        logging.info("REVIEW LIMIT EXCEEDED.")
        return "end"

    if plan.review.is_valid:
        logging.info("REVIEW SUCCESSFUL.")
        return "executor"
    else:
        logging.info("REVIEW FAILED.")
        return "planner"

def tool_execute(state):
    logging.info(f"tool_execute routing")
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        logging.info(f"tool_execute")
        return "tools"

    executor = state.get("executor")
    if not executor or not executor.step:
        return END

    return END

def reflection_routing(state):
    logging.info(f"reflection routing")

    if state['step_pointer'] > len(state['plan'].plan.steps):
        return END

    
    if state['executor'].status == 'SUCCESS':
        logging.info(f"SUCCESS execution. Next step")
        return 'tool_expert'

    logging.info(f"ERROR while execution. Replannning") # TODO debug only
    return 'error'

def workflow(
        planner_node: PlanningExpert,
        reviewer_node: ConstraintExpert,
        executor_node: ExecutorExpert,
        tool_expert: ToolExpert,
        tool_node: ToolNode,
        reflection_expert: ReflectionExpert,
) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node.node)
    workflow.add_node("reviewer", reviewer_node.node)
    workflow.add_node("executor", executor_node.node)
    workflow.add_node("tool", tool_expert.node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("reflection_expert", reflection_expert.node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        plan_review_loop,
        {
            "planner": "planner",
            "executor": "executor",
        }
    )

    workflow.add_edge("executor", "tool")
    
    workflow.add_conditional_edges(
        "tool",
        tool_execute,
        {
            "tools": "tool_node",
            #END: END, # TODO do we need it? any other logic??
        }
    )

    workflow.add_edge("tool_node", "reflection_expert")

    workflow.add_conditional_edges(
    "reflection_expert",
    reflection_routing,
    {
        "tool_expert": "tool",
        'error': END, # TODO END -> PlanningExpert
        END: END
    }
)

    return workflow


if __name__ == "__main__":
    def meta_callback(meta, started):
        pass


    with open(f"{PATH_PREFFIX}credentials.yml", "r") as f:
        config = yaml.safe_load(f)

    base_url = config["HOST_URL"]
    model_name = config['MODEL_NAME']

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    llm = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="",
        temperature=0.0,
        max_tokens=1000,
    )
    planning_expert = PlanningExpert(
        persona_path=f"{PATH_PREFFIX}prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )
    constraint_expert = ConstraintExpert(
        persona_path=f"{PATH_PREFFIX}prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )
    execution_expert = ExecutorExpert(
        persona_path=f"{PATH_PREFFIX}prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc=tools_desc_str,
        callback=meta_callback,
    )

    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    tool_expert = ToolExpert(
        persona_path=f"{PATH_PREFFIX}prompts/oss-20b-synthetic-persona",
        llm=llm_with_tools,
        tools=[report_task_completion],
        callback=meta_callback,
    )
    reflection_expert = ReflectionExpert()

    app = workflow(
        planning_expert, 
        constraint_expert,
        execution_expert,
        tool_expert,
        tool_node,
        reflection_expert,
        ).compile()

    png_bytes = app.get_graph().draw_mermaid_png()

    with open('planning_constraint_graph_final.png', "wb") as f:
        f.write(png_bytes)

    config = RunnableConfig(recursion_limit=50)
    app.invoke(

    {
        'input_task': 'Count characters in word raspberry',
        'messages': [],
        'step_pointer': 0
    }
    , config)
