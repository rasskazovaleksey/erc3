from typing import TypedDict, Optional, List

from pydantic import BaseModel

from erc.experts.schemas import ExecutionPlan, ConstraintExpertOutput, PlanStep
from typing import Annotated, List
import operator


class Plan(BaseModel):
    plan: ExecutionPlan
    is_validated: bool
    validation_attempts: Optional[int]
    review: Optional[ConstraintExpertOutput]

class ExecutionTool(BaseModel):
    step: PlanStep
    tool: str
    status: str

class AgentState(TypedDict):
    input_task: str
    plan: Optional[Plan]
    executor: Optional[ExecutionTool]
    step_pointer:int #TODO ADD MAX STEPS (END condition after last step)
    messages: Annotated[List, operator.add]