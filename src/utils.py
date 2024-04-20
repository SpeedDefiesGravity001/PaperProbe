from langchain.chat_models import ChatOpenAI
import os
import logging
from logger_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

def get_model(model_name):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if model_name == "openai":
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    return model


    












