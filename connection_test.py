import textwrap

from erc3 import ERC3
import yaml

def get_erc3_key() -> str:
    with open("credentials.yml", "r") as f:
        config = yaml.safe_load(f)
    return config["ERC3_API_KEY"]

if __name__ == "__main__":
    core = ERC3(key=get_erc3_key())

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
        # start the task
        core.start_task(task)
        try:
            store_api = core.get_demo_client(task)
            secret = store_api.get_secret()
            print(f"Received secret: {secret.secret}")
        except Exception as e:
            print(e)
        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")

    core.submit_session(res.session_id)