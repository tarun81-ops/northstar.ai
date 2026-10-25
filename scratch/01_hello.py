"""Lesson 1: a single model call with a system + human message.
Run from the project root:  python -m scratch.01_hello
"""
from langchain_core.messages import HumanMessage, SystemMessage

from northstar.llm import get_llm

model = get_llm()

system_message = SystemMessage(
    content="You are a helpful personal assistant for students to help them track their academic progress, manage their coursework, provide academic support and help them find their niche then do research onto that area."
)

human_message = HumanMessage(
    content="Hello, I need help organizing my coursework for this semester."
)

response = model.invoke([system_message, human_message])
print(response.content)
