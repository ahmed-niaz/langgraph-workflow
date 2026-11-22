from typing import Any, Literal, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph


# Agent state
class AgentState(TypedDict):
    number_one: int
    number_two: int
    operation: str
    final_number: NotRequired[int]


def adder(state: AgentState) -> dict[str, Any]:
    """This node adds the two numbers."""
    return {"final_number": state["number_one"] + state["number_two"]}


def substractor(state: AgentState) -> dict[str, Any]:
    """This node subtracts the two numbers."""
    return {"final_number": state["number_one"] - state["number_two"]}


# Conditional routing function
RouteChoice = Literal["additional_operation", "subtraction_operation"]


def decide_next_node(state: AgentState) -> RouteChoice:
    """Select the next node based on the operation symbol."""
    if state["operation"] == "+":
        return "additional_operation"
    if state["operation"] == "-":
        return "subtraction_operation"
    raise ValueError(f"Unsupported operation: {state.get('operation')}")


# Build the state graph
graph = StateGraph(AgentState)

# 1. Register nodes (including explicit passthrough router)
graph.add_node("router", lambda state: state)
graph.add_node("add_node", adder)
graph.add_node("subtract_node", substractor)

# 2. Connect START to the router node
graph.add_edge(START, "router")

# 3. Branch conditionally from router based on decide_next_node output
graph.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "additional_operation": "add_node",
        "subtraction_operation": "subtract_node",
    },
)

# 4. Connect both worker nodes to END
graph.add_edge("add_node", END)
graph.add_edge("subtract_node", END)

# Compile the graph
app = graph.compile()

if __name__ == "__main__":
    test_input: AgentState = {
        "number_one": 12,
        "operation": "+",
        "number_two": 34,
    }

    result = app.invoke(cast(Any, test_input))
    print(f"Result: {result['final_number']}")
