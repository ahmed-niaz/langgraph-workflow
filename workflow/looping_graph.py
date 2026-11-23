import random
from typing import Any, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    name: str
    number: list[int]
    counter: int


# define node
def greeting_node(state: AgentState) -> AgentState:
    """Greeting node which says hi to the person"""
    state["name"] = f"Hi there, {state['name']}"
    state["counter"] = 0
    return state


def random_node(state: AgentState) -> AgentState:
    """Generate a random number form 0 to 10"""
    state["number"].append(random.randint(0, 10))
    state["counter"] += 1
    return state


LoopChoice = Literal["loop", "exit"]


def should_continue(state: AgentState) -> LoopChoice:
    """Function to decide what to do next."""
    if state["counter"] < 5:
        print("Entering Loop: ", state["counter"])
        return "loop"
    else:
        return "exit"


# Build the state graph
graph = StateGraph(AgentState)

graph.add_node("greeting", greeting_node)
graph.add_node("random_number", random_node)

graph.add_edge(START, "greeting")
graph.add_edge("greeting", "random_number")

graph.add_conditional_edges(
    "random_number",
    should_continue,
    {
        "loop": "random_number",  # loop back to the same node
        "exit": END,
    },
)

# Compile the graph
app = graph.compile()

if __name__ == "__main__":
    test_input: AgentState = {
        "name": "Nest",
        "number": [7],
        "counter": 0,
    }

    result = app.invoke(cast(Any, test_input))
    print(f"Result: {result['number']}")
