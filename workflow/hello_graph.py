from typing import Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph


# Agent state
class AgentState(TypedDict):
    message: str


# Define node: functions or operations that perform specific tasks in the graph
def greeting_node(state: AgentState) -> dict[str, Any]:
    """Simple node that adds greeting message to the state."""
    greeting = f"Hey {state['message']}, how is your day going? "
    return {"message": greeting}


# Build the state graph
graph = StateGraph(AgentState)

# Add node and edges using modern LangGraph START and END
graph.add_node("greeter", greeting_node)
graph.add_edge(START, "greeter")
graph.add_edge("greeter", END)

# Compile the graph
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke(cast(Any, {"message": "Nest"}))
    print(result["message"])
