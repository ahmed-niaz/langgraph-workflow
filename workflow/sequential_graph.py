from typing import Any, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph


# Agent state
class AgentState(TypedDict):
    name: str
    age: int | str
    final: NotRequired[str]


# Define node:
def node_one(state: AgentState) -> AgentState:
    state["final"] = f"hi, {state['name']}"
    return state


def node_two(state: AgentState) -> AgentState:
    state["final"] = f"{state['final']}. You are {state['age']} years old."
    return state


# Build the state graph
graph = StateGraph(AgentState)

graph.add_node("node_one", node_one)
graph.add_node("node_two", node_two)


graph.add_edge(START, "node_one")
graph.add_edge("node_one", "node_two")
graph.add_edge("node_two", END)

# Compile the graph
app = graph.compile()


if __name__ == "__main__":
    # Test with different moods to see branching in action
    test_input: AgentState = {
        "name": "Nest",
        "age": 26,
    }

    result = app.invoke(cast(Any, test_input))
    print(result["final"])
