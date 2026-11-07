from ..core.config import settings

from langchain_openai import ChatOpenAI
class LLMClient:
    def __init__(self):
        self.base_url = settings.OPENAI_BASE_URL
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        
    def get_llm(self):
        return ChatOpenAI(
            model_name=self.model,
            base_url=self.base_url,
            api_key=self.api_key
        )
        
llm_client=LLMClient()
