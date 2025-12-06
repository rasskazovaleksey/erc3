from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.persona import PersonaProvider


class FeedbackExpert(BaseExpert):

    def __init__(self, persona_path, tool_desc: str, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("feedback_expert", persona_path)
        self.tools_desc = tool_desc
        # self.llm = llm.with_structured_output(ConstraintExpertOutput)
        self.callback = callback
