from typing import TypedDict, Optional, List

from pydantic import BaseModel

from erc.experts.schemas import ExecutionPlan, ConstraintExpertOutput, PlanStep


class Plan(BaseModel):
    plan: ExecutionPlan
    is_validated: bool
    validation_attempts: int
    review: Optional[ConstraintExpertOutput]

class ExecutionTool(BaseModel):
    step: PlanStep
    tool: str

class AgentState(TypedDict):
    input_task: str
    plan: Optional[Plan]
    executor: List[ExecutionTool]
