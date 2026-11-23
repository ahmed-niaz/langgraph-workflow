import os
from typing import Any, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# Load environment variables (.env searched upward automatically)
load_dotenv(find_dotenv())


def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
    )


# Initialize LLM instance
llm = get_llm()


class AgentState(TypedDict):
    messages: list[HumanMessage]


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])

    # Extract only the clean text
    if isinstance(response.content, list) and response.content:
        first = response.content[0]
        if isinstance(first, dict):
            text = str(first.get("text", ""))
        else:
            text = str(first)
    else:
        text = str(response.content)

    print(f"\nAi: {text}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

if __name__ == "__main__":
    while True:
        user_input = input("Enter (or 'exit' to quit): ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        input_message = HumanMessage(content=user_input)
        agent.invoke(cast(Any, {"messages": [input_message]}))
