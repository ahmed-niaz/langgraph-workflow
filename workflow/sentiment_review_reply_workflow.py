import os
from typing import Any, Literal, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

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


# ---------------------------------------------------------------------------
# Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description=(
            "The sentiment of the review: "
            "positive, negative, or neutral."
        )
    )


class DiagnosisSchema(BaseModel):
    issue_type: Literal["billing", "technical", "account", "other"] = Field(
        description="The type of issue reported in the review."
    )
    tone: Literal["formal", "informal"] = Field(
        description="The recommended tone for the response."
    )
    urgency: Literal["high", "medium", "low"] = Field(
        description="The urgency of the response."
    )


# Structured model instances
sentiment_classifier = llm.with_structured_output(SentimentSchema)
diagnosis_analyzer = llm.with_structured_output(DiagnosisSchema)


# ---------------------------------------------------------------------------
# LangGraph State Schema
# ---------------------------------------------------------------------------
class ReviewAnalysisState(TypedDict, total=False):
    review: str
    sentiment: Literal["positive", "negative", "neutral"]
    diagnosis: dict[str, Any]
    response: str


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------
def find_sentiment(state: ReviewAnalysisState) -> dict[str, Any]:
    """Analyze customer review and classify its sentiment."""
    review = state.get("review", "")
    prompt = (
        "Analyze the customer review and determine whether its sentiment "
        "is positive, negative, or neutral:\n\n"
        f"Review: {review}"
    )
    result = sentiment_classifier.invoke(prompt)
    if isinstance(result, SentimentSchema):
        sentiment = result.sentiment
    elif isinstance(result, dict):
        sentiment = result.get("sentiment", "neutral")
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment}


def positive_response(state: ReviewAnalysisState) -> dict[str, Any]:
    """Generate an appreciative reply for positive customer reviews."""
    review = state.get("review", "")
    prompt = (
        "You are a customer success representative. "
        "A customer left a positive review:\n"
        f"'{review}'\n\n"
        "Write a warm, polite, and enthusiastic response thanking them "
        "for their business."
    )
    res = llm.invoke(prompt)
    return {"response": extract_text(res.content)}


def neutral_response(state: ReviewAnalysisState) -> dict[str, Any]:
    """Generate a helpful, balanced reply for neutral customer reviews."""
    review = state.get("review", "")
    prompt = (
        "You are a customer support agent. "
        "A customer left a neutral review:\n"
        f"'{review}'\n\n"
        "Write a courteous, professional response acknowledging their "
        "feedback and kindly asking how we can improve their experience."
    )
    res = llm.invoke(prompt)
    return {"response": extract_text(res.content)}


def run_diagnosis(state: ReviewAnalysisState) -> dict[str, Any]:
    """Diagnose negative reviews to determine category, tone, and urgency."""
    review = state.get("review", "")
    prompt = (
        "Diagnose this negative customer review to identify the issue "
        "type, tone, and urgency:\n\n"
        f"Review: {review}"
    )
    result = diagnosis_analyzer.invoke(prompt)
    if isinstance(result, DiagnosisSchema):
        diagnosis_dict = result.model_dump()
    elif isinstance(result, dict):
        diagnosis_dict = result
    else:
        diagnosis_dict = {}
    return {"diagnosis": diagnosis_dict}


def negative_response(state: ReviewAnalysisState) -> dict[str, Any]:
    """Generate an empathetic resolution reply tailored to diagnosis."""
    review = state.get("review", "")
    diagnosis = state.get("diagnosis", {})
    issue_type = diagnosis.get("issue_type", "general")
    urgency = diagnosis.get("urgency", "medium")
    tone = diagnosis.get("tone", "formal")

    prompt = (
        "You are a customer resolution specialist.\n"
        f"Customer Review: '{review}'\n"
        f"Diagnosed Issue: {issue_type}\n"
        f"Urgency Level: {urgency}\n"
        f"Tone Required: {tone}\n\n"
        "Write a sincere, empathetic, and action-oriented response "
        "addressing their specific issue and providing a clear path "
        "to resolution."
    )
    res = llm.invoke(prompt)
    return {"response": extract_text(res.content)}


# ---------------------------------------------------------------------------
# Conditional Edge Router
# ---------------------------------------------------------------------------
RouteDestination = Literal[
    "positive_response",
    "neutral_response",
    "run_diagnosis",
]


def check_sentiment(state: ReviewAnalysisState) -> RouteDestination:
    """Route review to the appropriate handler based on sentiment."""
    sentiment = state.get("sentiment")
    if sentiment == "positive":
        return "positive_response"
    if sentiment == "neutral":
        return "neutral_response"
    return "run_diagnosis"


# ---------------------------------------------------------------------------
# StateGraph Construction
# ---------------------------------------------------------------------------
graph = StateGraph(ReviewAnalysisState)

# Add nodes
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("neutral_response", neutral_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

# Add edges
graph.add_edge(START, "find_sentiment")

# Conditional branch from find_sentiment
graph.add_conditional_edges(
    "find_sentiment",
    check_sentiment,
    {
        "positive_response": "positive_response",
        "neutral_response": "neutral_response",
        "run_diagnosis": "run_diagnosis",
    },
)

# Negative pathway: diagnosis -> response
graph.add_edge("run_diagnosis", "negative_response")

# Terminal edges
graph.add_edge("positive_response", END)
graph.add_edge("neutral_response", END)
graph.add_edge("negative_response", END)

# Compile into a runnable workflow
workflow = graph.compile()


# ---------------------------------------------------------------------------
# Demonstration / Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_reviews = [
        (
            "The software works great! Customer support was fast "
            "and helped me setup in minutes."
        ),
        (
            "The package arrived on the scheduled date. "
            "Quality is average, meets expectations."
        ),
        (
            "I am very disappointed with the service. "
            "The product arrived late and was damaged. "
            "I want a refund immediately!"
        ),
    ]

    for idx, review_text in enumerate(test_reviews, start=1):
        print(f"\n{'=' * 70}")
        print(f"TEST CASE {idx}:")
        print(f'Review: "{review_text}"')
        print(f"{'=' * 70}")

        input_state: ReviewAnalysisState = {"review": review_text}
        result = workflow.invoke(cast(Any, input_state))

        print(f"Classified Sentiment: {result.get('sentiment')}")
        if result.get("diagnosis"):
            print(f"Diagnosis Details:    {result.get('diagnosis')}")
        print(f"\nGenerated Reply:\n{result.get('response')}\n")
