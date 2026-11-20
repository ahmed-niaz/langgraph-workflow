from typing import Any, NotRequired, TypedDict, cast

from langgraph.graph import END, START, StateGraph


# agent state
class AgentState(TypedDict):
    values: list[int]  # REQUIRED: Caller MUST provide this
    name: str  # REQUIRED: Caller MUST provide this
    result: NotRequired[str]  # OPTIONAL: Node will populate this later


# Define Node 1
def process_values(state: AgentState) -> AgentState:
    """this func process hangles multiple differnt input"""
    total = sum(state["values"])
    state["result"] = f"Hi there {state['name']}! Your sum = {total}"
    return state


# create the graph
graph = StateGraph(AgentState)


# 2 params [ name, aciton]
graph.add_node("processor", process_values)

graph.add_edge(START, "processor")
graph.add_edge("processor", END)


# Compile the graph
app = graph.compile()

if __name__ == "__main__":
    test_input: AgentState = {
        "values": [1, 2, 3, 4],
        "name": "Nest",
    }

    result = app.invoke(cast(Any, test_input))
    print(result["result"])
