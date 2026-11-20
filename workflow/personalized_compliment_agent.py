import os
from typing import Any, Literal, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
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


def extract_text(content: Any) -> str:
    """Extract clean string text from LangChain / Google GenAI response."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


MoodType = Literal["tired", "excited", "curious"]


# ---------------------------------------------------------------------------
# 1. Agent State Definition (Tracks input and outputs through the workflow)
# ---------------------------------------------------------------------------
class ComplimentState(TypedDict, total=False):
    name: str
    topic: str
    mood: MoodType
    compliment: str
    quote: str
    final_message: str


# ---------------------------------------------------------------------------
# 2. Graph Nodes
# ---------------------------------------------------------------------------
def energizing_compliment(state: ComplimentState) -> dict[str, Any]:
    """Generates a warm, supportive compliment when feeling tired."""
    name = state.get("name", "Friend")
    topic = state.get("topic", "learning")
    prompt = (
        f"The user '{name}' is feeling tired while working on '{topic}'.\n"
        "Write a warm, comforting, and encouraging compliment (2 sentences) "
        "reminding them that rest is part of progress."
    )
    res = llm.invoke(prompt)
    return {"compliment": extract_text(res.content)}


def celebratory_compliment(state: ComplimentState) -> dict[str, Any]:
    """Generates a high-energy, celebratory compliment when excited."""
    name = state.get("name", "Friend")
    topic = state.get("topic", "learning")
    prompt = (
        f"The user '{name}' is feeling excited while mastering '{topic}'.\n"
        "Write a high-energy, celebratory compliment (2 sentences) cheering "
        "on their momentum and enthusiasm."
    )
    res = llm.invoke(prompt)
    return {"compliment": extract_text(res.content)}


def curiosity_compliment(state: ComplimentState) -> dict[str, Any]:
    """Generates a thoughtful, inspiring compliment when feeling curious."""
    name = state.get("name", "Friend")
    topic = state.get("topic", "learning")
    prompt = (
        f"The user '{name}' is feeling curious exploring '{topic}'.\n"
        "Write an inspiring compliment (2 sentences) praising their deep "
        "curiosity and passion for discovery."
    )
    res = llm.invoke(prompt)
    return {"compliment": extract_text(res.content)}


def add_quote_node(state: ComplimentState) -> dict[str, Any]:
    """Demonstrates a convergence (fan-in) node: adds an inspiring quote."""
    topic = state.get("topic", "general").lower()
    quotes = {
        "langgraph": (
            '"Small composable graphs build intelligent systems." '
            "— Agent Architect"
        ),
        "coding": (
            '"The only way to learn a new tool is to build with it." '
            "— Dennis Ritchie"
        ),
        "general": (
            '"It does not matter how slowly you go as long as you '
            'do not stop." — Confucius'
        ),
    }
    selected_quote = quotes.get(topic, quotes["general"])
    return {"quote": selected_quote}


def format_card_node(state: ComplimentState) -> dict[str, Any]:
    """Formats accumulated state into a final greeting card message."""
    name = state.get("name", "Friend")
    topic = state.get("topic", "learning")
    mood = state.get("mood", "curious")
    compliment = state.get("compliment", "")
    quote = state.get("quote", "")

    card = (
        f"✨ PERSONALIZED COMPLIMENT FOR {name.upper()} ✨\n"
        f"Topic: {topic} | Current Vibe: {mood}\n\n"
        f"{compliment}\n\n"
        f"💡 Daily Inspiration:\n{quote}"
    )
    return {"final_message": card}


# ---------------------------------------------------------------------------
# 3. Conditional Routing Function
# ---------------------------------------------------------------------------
RouteDestination = Literal[
    "energizing_compliment",
    "celebratory_compliment",
    "curiosity_compliment",
]


def route_by_mood(state: ComplimentState) -> RouteDestination:
    """Decides which compliment node to execute based on user mood."""
    mood = state.get("mood", "curious")
    if mood == "tired":
        return "energizing_compliment"
    if mood == "excited":
        return "celebratory_compliment"
    return "curiosity_compliment"


# ---------------------------------------------------------------------------
# 4. StateGraph Assembly
# ---------------------------------------------------------------------------
graph = StateGraph(ComplimentState)

# Add all nodes
graph.add_node("energizing_compliment", energizing_compliment)
graph.add_node("celebratory_compliment", celebratory_compliment)
graph.add_node("curiosity_compliment", curiosity_compliment)
graph.add_node("add_quote", add_quote_node)
graph.add_node("format_card", format_card_node)

# Conditional routing from START: routes directly to one of the 3 generators
graph.add_conditional_edges(
    START,
    route_by_mood,
    {
        "energizing_compliment": "energizing_compliment",
        "celebratory_compliment": "celebratory_compliment",
        "curiosity_compliment": "curiosity_compliment",
    },
)

# Fan-in (convergence): all 3 branches flow into add_quote
graph.add_edge("energizing_compliment", "add_quote")
graph.add_edge("celebratory_compliment", "add_quote")
graph.add_edge("curiosity_compliment", "add_quote")

# Sequence: add_quote -> format_card -> END
graph.add_edge("add_quote", "format_card")
graph.add_edge("format_card", END)

# Compile into runnable workflow
app = graph.compile()


# ---------------------------------------------------------------------------
# 5. Execution Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test with different moods to see branching in action
    test_input: ComplimentState = {
        "name": "Nest",
        "topic": "LangGraph",
        "mood": "tired",  # Try changing to "excited" or "curious"
    }

    result = app.invoke(cast(Any, test_input))
    print(result["final_message"])
