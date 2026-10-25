"""Lesson 3: structured output. The model must return data matching a Pydantic schema,
so the app gets real Python objects (ready for the database) instead of free text.
Run from the project root:  python -m scratch.03_structured
"""
from datetime import date

from pydantic import BaseModel, Field

from northstar.llm import get_llm


class Milestone(BaseModel):
    title: str
    due_date: date
    completion_criteria: str = Field(description="Something observable, e.g. 'solves 8/10 past-paper questions'")


class MilestoneList(BaseModel):
    milestones: list[Milestone] = Field(min_length=4, max_length=8)


structured_llm = get_llm().with_structured_output(MilestoneList)

result = structured_llm.invoke(
    f"Today is {date.today()}. A grade 10 student with 2 free hours on weekdays wants to "
    "'score 85% in maths and finish a line-follower robot by 15 December'. Create 4-8 milestones."
)

# result is a MilestoneList object, not a string.
for m in result.milestones:
    print(f"{m.due_date}  {m.title}  ->  {m.completion_criteria}")

# Python, not the model, checks the dates make sense (the habit you'll use in Phase 4).
dates = [m.due_date for m in result.milestones]
assert dates == sorted(dates), "milestone dates are not in order"
print("OK: got", len(result.milestones), "milestones with ascending dates")
