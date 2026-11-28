import os
from typing import Annotated, Any, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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


# AgentState with add_messages reducer so messages accumulate automatically
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def process(state: AgentState) -> dict[str, Any]:
    # Invoke LLM with complete conversation history from memory
    response = llm.invoke(state["messages"])

    # Extract clean text from response
    if isinstance(response.content, list) and response.content:
        first = response.content[0]
        if isinstance(first, dict):
            text = str(first.get("text", ""))
        else:
            text = str(first)
    else:
        text = str(response.content)

    print(f"\nAi: {text}\n")

    # Return the new AI message; add_messages appends it to memory
    return {"messages": [AIMessage(content=text)]}


# Build the state graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# In-memory checkpointer enables real persistent memory across invocations
memory = MemorySaver()
agent = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    # thread_id identifies this specific conversation session
    config = {"configurable": {"thread_id": "1"}}

    print("--- Chatbot with Memory Initialized (type 'exit' to quit) ---")
    while True:
        user_input = input("Enter: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break

        # Pass ONLY the new message; LangGraph automatically loads history!
        agent.invoke(
            cast(
                Any,
                {"messages": [HumanMessage(content=user_input)]},
            ),
            config=cast(Any, config),
        )

    # Save conversation log from memory checkpoint
    checkpoint_state = agent.get_state(cast(Any, config))
    conversation_history = checkpoint_state.values.get("messages", [])

    with open("logging.txt", "w", encoding="utf-8") as file:
        file.write("Your Conversation Log:\n\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n\n")
        file.write("End of Conversation\n")

    print("Conversation saved to logging.txt")
