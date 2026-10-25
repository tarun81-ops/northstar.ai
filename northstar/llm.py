"""The one place the app creates a chat model. Switch provider via NORTHSTAR_MODEL."""
from langchain.chat_models import init_chat_model

from northstar import config


def get_llm(**kwargs):
    return init_chat_model(config.MODEL, **kwargs)
