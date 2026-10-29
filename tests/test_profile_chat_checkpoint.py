from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.orm import sessionmaker

from northstar.agents import profile_agent
from northstar.ai.catalog import ModelEntry, catalog
from northstar.db.session import init_db, make_engine


class RateLimitedError(Exception):
    status_code = 429


class ScriptedToolModel(BaseChatModel):
    responses: list
    response_index: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.response_index]
        self.response_index += 1
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def _update_call(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "update_student_profile",
                "args": {
                    "alias": "Meera",
                    "age_band": "14-16",
                    "grade": "Grade 9",
                    "interests": ["Art"],
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_model_switch_resumes_checkpoint_without_repeating_profile_write(monkeypatch):
    engine = make_engine("sqlite://")
    init_db(engine)
    session_factory = sessionmaker(engine)
    first_model = ScriptedToolModel(
        responses=[_update_call("call-1"), RateLimitedError("rate limited")]
    )
    fallback_model = ScriptedToolModel(
        responses=[_update_call("call-2"), AIMessage("Profile updated.")]
    )
    first_model_id = "openrouter:fixture/first:free"
    fallback_model_id = "openrouter:fixture/second:free"
    free_models = [
        ModelEntry(
            provider="openrouter",
            model_id=model_id,
            name=model_id,
            verified=False,
            source="catalog",
            pricing={"prompt": "0", "completion": "0"},
            supports_tool_calling=True,
            supported_parameters=frozenset({"tools"}),
        )
        for model_id in (first_model_id, fallback_model_id)
    ]
    monkeypatch.setattr(catalog, "get_openrouter_models", lambda: free_models)
    llm_calls = []

    def fake_get_llm(task, *, model=None, **kwargs):
        llm_calls.append((task, model, kwargs))
        return fallback_model if model == fallback_model_id else first_model

    monkeypatch.setattr(profile_agent, "get_llm", fake_get_llm)

    from northstar.services.profile import save_profile as original_save
    import northstar.tools.profile_tools as profile_tools

    saves = []

    def counted_save(session, profile, student_id=None):
        saves.append(student_id)
        return original_save(session, profile, student_id)

    monkeypatch.setattr(profile_tools, "save_profile", counted_save)
    agent, state = profile_agent.build_profile_agent(session_factory=session_factory)
    checkpointer = state["_checkpointer"]
    config = {"configurable": {"thread_id": "same-chat"}}

    try:
        agent.invoke(
            {"messages": [HumanMessage("Update my profile", id="pending-user-message")]},
            config=config,
        )
    except RateLimitedError:
        pass
    else:
        raise AssertionError("Expected the first provider to be rate-limited")

    assert state["student_id"] is not None
    assert [message.type for message in agent.get_state(config).values["messages"]] == [
        "human",
        "ai",
        "tool",
    ]

    fallback_agent, _ = profile_agent.build_profile_agent(
        state["student_id"],
        session_factory,
        model=fallback_model_id,
        tools=state["_tools"],
        state=state,
        checkpointer=checkpointer,
    )
    result = fallback_agent.invoke(None, config=config)

    assert result["messages"][-1].content == "Profile updated."
    assert saves == [None]
    assert all(task is None for task, _, _ in llm_calls)
    assert all(kwargs["free_only"] for _, _, kwargs in llm_calls)
