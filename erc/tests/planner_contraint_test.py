import logging
import sys

import yaml
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from erc.experts.constraint import ConstraintExpert
from erc.experts.planning import PlanningExpert
from erc.state import AgentState


def executor_node(state: AgentState):
    logging.info("MOCK EXECUTOR NODE")
    return state


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


def workflow(
        planner_node: PlanningExpert,
        reviewer_node: ConstraintExpert,
) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node.node)
    workflow.add_node("reviewer", reviewer_node.node)
    workflow.add_node("executor", executor_node)

    workflow.set_entry_point("planner")
    workflow.set_finish_point("executor")

    workflow.add_edge("planner", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        plan_review_loop,
        {
            "planner": "planner",
            "executor": "executor",
        }
    )

    return workflow


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
    planning_expert = PlanningExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )
    constraint_expert = ConstraintExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc="""
            report_task_completion: Reports that the task has been completed successfully.
        """,
        callback=meta_callback,
    )

    app = workflow(planning_expert, constraint_expert).compile()

    png_bytes = app.get_graph().draw_mermaid_png()

    with open('planning_constraint_graph.png', "wb") as f:
        f.write(png_bytes)

    config = RunnableConfig(recursion_limit=50)
    app.invoke({
        "input_task": "Count characters in word raspberry",
    }, config)
