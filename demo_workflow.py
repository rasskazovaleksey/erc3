import logging
import sys
import textwrap

import yaml
from erc3 import ERC3, DemoClient
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, render_text_description
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from erc.experts.constraint import ConstraintExpert
from erc.experts.planning import PlanningExpert
from erc.experts.tool import ExecutorExpert
from erc.workflow import workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

DEMO_API: DemoClient = None  # TODO: think of better injection
CURRENT_TASK = None  # TODO: think of better injection


@tool
def provide_answer(answer: str) -> str:  # TODO: task id passed though LLM, this is bad, MB shall go into state
    """
    Provide an answer for the current task.
    Args:
        answer (str): The answer for the current task.
    """
    print("<----- report_task_completion")
    resp = DEMO_API.provide_answer(CURRENT_TASK, answer)
    print(f"RESPONSE: {resp}")
    return resp  # TODO: this is example only, course you hardcoded SUCCESS


@tool
def get_secret() -> str:
    """Get the secret value for the current task."""
    print("<----- get_secret")
    resp = DEMO_API.get_secret()
    return resp.value

TOOLS = [provide_answer, get_secret]


def get_erc3_key() -> str:
    with open("credentials.yml", "r") as f:
        config = yaml.safe_load(f)
    return config["ERC3_API_KEY"]


# def create_workflow(meta_callback, tools):
#     llm = ChatOpenAI(
#         model="oss-20b",
#         base_url="http://localhost:8080/v1",
#         api_key="",
#         temperature=0.0,
#         max_tokens=1000,
#     )
#     p = PlanningExpert(
#         persona_path="prompts/oss-20b-synthetic-persona",
#         llm=llm,
#         tools=[provide_answer, get_secret],
#         tool_desc="""
#             provide_answer: Reports that the task has been completed successfully. Final task.
#             get_secret: Gets the secret value for the current task.
#         """,
#         callback=meta_callback,
#     )
#     c = ConstraintExpert(
#         persona_path="prompts/oss-20b-synthetic-persona",
#         llm=llm,
#         tools=[provide_answer, get_secret],
#         tool_desc="""
#             provide_answer: Reports that the task has been completed successfully. Final task.
#             get_secret: Gets the secret value for the current task.
#         """,
#         callback=meta_callback,
#     )
#     e = ExecutorExpers(
#         persona_path="prompts/oss-20b-synthetic-persona",
#         llm=llm,
#         tools=[provide_answer, get_secret],
#         tool_desc="""
#             provide_answer: Reports that the task has been completed successfully. Final task.
#             get_secret: Gets the secret value for the current task.
#         """,
#         callback=meta_callback,
#     )

#     llm_with_tools = llm.bind_tools(tools)
#     tool_node = ToolNode(tools)

#     return workflow(p, c, tool_node)

def create_workflow(meta_callback, tools):

    llm = ChatOpenAI(
        model="oss-20b",
        base_url="http://localhost:8080/v1",
        api_key="",
        temperature=0.0,
        request_timeout=120.0
    )
    
    llm_with_tools = llm.bind_tools(tools)
    tools_desc_str = render_text_description(tools)
    
    p = PlanningExpert(
        persona_path="prompts/oss-20b-synthetic-persona",
        llm=llm_with_tools,
        tools=tools, 
        tool_desc=tools_desc_str,
        callback=meta_callback,
    )
    
    c = ConstraintExpert(
        persona_path="prompts/oss-20b-synthetic-persona",
        llm=llm_with_tools,
        tools=tools,
        tool_desc=tools_desc_str,
        callback=meta_callback,
    )
    
    e = ExecutorExpert(
        persona_path="prompts/oss-20b-synthetic-persona",
        llm=llm_with_tools, 
        tools=tools,
        tool_desc=tools_desc_str,
        callback=meta_callback,
    )

    tool_node_instance = ToolNode(tools)

    return workflow(p, c, e, tool_node_instance)

if __name__ == "__main__":
    core = ERC3(key=get_erc3_key())


    def meta_callback(meta, started):
        print(f"METADATA: {meta}")
        # core.log_llm(
        #     task_id=CURRENT_TASK,
        #     model="gpt-oss-20b", #TODO: EXTRANCT CORRECTLY from meta # must match slug from OpenRouter
        #     duration_sec=time.time() - started,
        #     usage=meta, # TODO: this is incorrect too
        # )


    app = create_workflow(meta_callback, tools=TOOLS).compile()

    logging.info("🚀 Starting Demo Agent...")
    # png_bytes = app.get_graph().draw_mermaid_png()
    #
    # with open('img.png', "wb") as f:
    #     f.write(png_bytes)

    config = RunnableConfig(recursion_limit=50)

    # Start session with metadata
    res = core.start_session(
        benchmark="demo",
        workspace="dev",
        name=f"connection test",
        architecture="none")

    status = core.session_status(res.session_id)
    print(f"Session has {len(status.tasks)} tasks")
    for task in status.tasks:
        print("=" * 40)
        print(f"Starting Task: {task.task_id} ({task.spec_id}): {task.task_text}")

        CURRENT_TASK = task.task_id
        # start the task
        core.start_task(task)
        try:
            demo_api = core.get_demo_client(task)
            DEMO_API = demo_api  # TODO: think of better injection

            app.invoke({
                "input_task": task.task_text,
                "consecutive_review_failures": 0,
                "history": [],
                "iterations": 0,
                "plan_is_valid": False,
            }, config)


        except Exception as e:
            print(e)

        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")

    core.submit_session(res.session_id)
