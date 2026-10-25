"""Lesson 2: a chain = prompt template | model | output parser (LCEL).
Run from the project root:  python -m scratch.02_chain
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from northstar.llm import get_llm

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are NorthStar, a warm, concise mentor for a {grade} student. Never shame the student."),
    ("human", "I have {minutes} free minutes today and I'm interested in {interest}. Suggest ONE small task."),
])

chain = prompt | get_llm() | StrOutputParser()

# The same chain, different inputs: templates are reusable.
print(chain.invoke({"grade": "grade 10", "minutes": 30, "interest": "robotics"}))
print("---")
print(chain.invoke({"grade": "first-year college", "minutes": 45, "interest": "creative writing"}))
