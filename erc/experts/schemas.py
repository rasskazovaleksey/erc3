# python
from typing import Union, Dict, Any, List, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    A single step in the execution plan.
    """
    tool_name: str = Field(None, description="Tool Name to be used in order to finish the task")
    arguments: Union[Dict[str, Any], str, None] = Field(None, description="Arguments to be passed to the tool")
    reasoning: str = Field(None, description="Reasoning to be shown when executing the task. Keep in short")
    summary: str = Field(None, description="Summary of the execution step")
    


class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(None, description="List of execution steps")


class ConstraintExpertOutput(BaseModel):
    is_valid: bool = Field(description="Indicates whether the current plan satisfies all constraints.")
    review_feedback: str = Field(description="Detailed feedback on any constraint violations or confirmations.")


class ExecutorExpertOutput(BaseModel):
    decision: Literal["tool", "code"] = Field(
        description="The decision made by the executor regarding the next action.")


class Feedback(BaseModel):
    reviewer: str = Field(description="Name or identifier of the reviewer providing the feedback.")
    comments: str = Field(description="Detailed comments or feedback from the reviewer.")
