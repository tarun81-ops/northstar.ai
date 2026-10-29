"""Profile Agent: a tool-calling loop (model -> tool -> model ...) that interviews the student
and writes the profile to the database through profile_tools.

    agent, state = build_profile_agent(student_id)
    result = agent.invoke({"messages": history})   # history = list of chat messages
    result["messages"]                            # full history incl. tool calls and the new reply
"""
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import MemorySaver

from northstar.ai.context_manager import trim_messages
from northstar.db.session import get_session
from northstar.ai.model_router import get_llm
from northstar.ai.model_factory import resolve_openrouter_free_model
from northstar.prompts import system_prompt
from northstar.tools.profile_tools import make_profile_tools


@before_model
def bound_conversation_context(state, runtime):
    """Keep model input bounded without splitting completed tool-call pairs."""
    messages = state["messages"]
    trimmed = trim_messages(messages, max_messages=40)
    if len(trimmed) == len(messages):
        return None
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *trimmed,
        ]
    }


def greeting(alias: str | None = None) -> str:
    """Fixed first message (no LLM call needed)."""
    if alias:
        return f"Welcome back, {alias}! Has anything changed - new interests, a different schedule, upcoming exams? Tell me and I'll update your profile."
    return ("Hi! I'm NorthStar, your study mentor. I'll ask a few quick questions so I can plan around your life. "
            "What nickname should I call you, and what grade or year are you in?")


def build_profile_agent(
    student_id: int | None = None,
    session_factory=get_session,
    *,
    model: str | None = None,
    tools=None,
    state=None,
    checkpointer=None,
):
    """Build the profile agent while allowing safe model replacement per chat."""
    if tools is None or state is None:
        tools, state = make_profile_tools(student_id, session_factory)
    if checkpointer is None:
        checkpointer = MemorySaver()
    model = resolve_openrouter_free_model(
        model,
        required_capabilities={"tool_calling"},
    )
    state["_tools"] = tools
    state["_checkpointer"] = checkpointer
    state["_model"] = model
    agent = create_agent(
        get_llm(
            None,
            model=model,
            required_capabilities={"tool_calling"},
            free_only=True,
            minimal_request=True,
        ),
        tools,
        system_prompt=system_prompt("profile"),
        middleware=[bound_conversation_context],
        checkpointer=checkpointer,
    )
    return agent, state
