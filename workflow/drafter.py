# Task:

# Our company is not working efficiently! We spend way too much time
# drafting documents and this needs to be fixed!

# For the company, you need to create an AI Agentic System that can
# speed up drafting documents, emails, etc. The AI Agentic System
# should have Human-AI Collaboration meaning the Human should be
# able to provide continuous feedback and the AI Agent should
# stop when the Human is happy with the draft. The system should also
# be fast and be able to save the drafts.

import os
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables (.env searched upward automatically)
load_dotenv(find_dotenv())

# global variable to store document content
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# two tools : update and save


# the llm model automatically passes the parameter for this tool.
@tool
def update(content: str) -> str:
    """Updated the document with provided content."""
    global document_content
    document_content = content

    return (
        "Document has been updated successfully! "
        f"The current content is:\n{document_content}"
    )


# the llm gives the file name as well.
@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except OSError as e:
        return f"Error saving document: {e!s}"


tools = [update, save]


def get_llm() -> Any:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
    ).bind_tools(tools)


# Initialize LLM instance
llm = get_llm()


def our_agent(state: AgentState) -> dict[str, Any]:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the
    user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool
      with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after
      modifications.

    The current document content is:{document_content}
    """
    )

    if not state["messages"]:
        prompt_intro = "I'm ready to help you update a document. "
        user_input = f"{prompt_intro}What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = ""
        while not user_input:
            user_input = input(
                "\nWhat would you like to do with the document? "
            ).strip()
        print(f"\n user: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = llm.invoke(all_messages)

    if isinstance(response.content, list) and response.content:
        first = response.content[0]
        if isinstance(first, dict):
            ai_text = str(first.get("text", ""))
        else:
            ai_text = str(first)
    else:
        ai_text = str(response.content)

    if ai_text:
        print(f"\n ai: {ai_text}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"using tools: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [user_message, response]}


# conditional edge
def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (
            isinstance(message, ToolMessage)
            and isinstance(message.content, str)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"  # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages: Sequence[BaseMessage]) -> None:
    """Function to print the messages in a more readable format."""
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n tool result: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent() -> None:
    print("\n ===== DRAFTER =====")

    state: AgentState = {"messages": []}

    for step in app.stream(cast(Any, state), stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
