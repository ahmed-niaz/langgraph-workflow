# ruff: noqa: N999
# Objectives:
# 1. Learn how to create Tools in LangGraph
# 2. How to create a ReAct [Reasoning acting graph] Graph
# 3. Work with different types of Messages such as ToolMessages
# 4. Test out robustness of our graph

from collections.abc import Sequence
import os
from typing import Annotated, Any, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables (.env searched upward automatically)
load_dotenv(find_dotenv())

# annotated - provide additional content without affecting the type itself
# sequence - handle state updates by adding new messages to chat history.

# Reducer function:
# Rule that controls how updates from nodes are combined with existing state.
# Tells us how to merge data into the current state.
# Without a reducer, updates would have replaced the existing value entirely.
# With a reducer, it works like an append method.
# add_messages is a reducer func to append without overriding.


class AgentState(TypedDict):
    # [datatype, metadata]
    messages: Annotated[Sequence[BaseMessage], add_messages]


# called decorator
@tool
def add(a: int, b: int) -> int:
    """This is an addition func that adds 2 numbers together"""
    return a + b


tools = [add]


def get_llm() -> Any:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
    ).bind_tools(tools)


# Initialize LLM instance
llm = get_llm()


def llm_call(state: AgentState) -> dict[str, Any]:
    system_prompt = SystemMessage(
        content=(
            "You are my AI assistant, please answer my query to "
            "the best of your ability."
        )
    )
    response = llm.invoke([system_prompt, *state["messages"]])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]

    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", llm_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")

graph.add_conditional_edges(
    "our_agent", should_continue, {"continue": "tools", "end": END}
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream: Any) -> None:
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    inputs = {"messages": [("user", "Add 3+4 , add 5+8")]}
    print_stream(app.stream(cast(Any, inputs), stream_mode="values"))
