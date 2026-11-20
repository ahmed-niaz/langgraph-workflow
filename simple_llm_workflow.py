import os
from typing import Any, TypedDict, cast

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


# Define the state schema
class LLMState(TypedDict, total=False):
    question: str
    answer: str
    min_length: int  # Minimum number of words (default: 40)
    max_length: int  # Maximum number of words (default: 90)
    word_count: int
    feedback: str
    revision_count: int
    is_valid: bool


# Node 1: Generate or refine answer
def generate_answer(state: LLMState) -> dict[str, Any]:
    question = state.get("question", "")
    min_len = state.get("min_length", 40)
    max_len = state.get("max_length", 90)
    feedback = state.get("feedback")

    if feedback:
        prompt = (
            "You are a helpful assistant.\n"
            f"Original Question: {question}\n\n"
            f"Your Previous Draft:\n{state.get('answer', '')}\n\n"
            f"Review Feedback:\n{feedback}\n\n"
            "Please revise your answer to directly answer the question "
            "and fix the feedback. "
            f"Strict constraint: The revised answer MUST be between "
            f"{min_len} and {max_len} words."
        )
    else:
        prompt = (
            "You are a helpful assistant. "
            "Please answer the following question:\n"
            f"{question}\n\n"
            f"Strict constraint: Your answer MUST be between {min_len} "
            f"and {max_len} words."
        )

    response = llm.invoke(prompt)
    answer_text = extract_text(response.content)
    return {"answer": answer_text}


# Node 2: Validate length criteria
def validate_length(state: LLMState) -> dict[str, Any]:
    answer = state.get("answer", "")
    min_len = state.get("min_length", 40)
    max_len = state.get("max_length", 90)
    rev_count = state.get("revision_count", 0)

    words = answer.split()
    word_count = len(words)

    if word_count < min_len:
        feedback = (
            f"The answer is only {word_count} words, which is below the "
            f"minimum required {min_len} words. "
            "Please elaborate with more helpful explanation and details "
            "to meet the minimum length."
        )
        is_valid = False
    elif word_count > max_len:
        feedback = (
            f"The answer is {word_count} words, which exceeds the maximum "
            f"allowed {max_len} words. "
            "Please condense and summarize the answer to fit under "
            f"{max_len} words."
        )
        is_valid = False
    else:
        feedback = ""
        is_valid = True

    return {
        "word_count": word_count,
        "feedback": feedback,
        "is_valid": is_valid,
        "revision_count": rev_count + 1,
    }


# Conditional routing function
def should_continue(state: LLMState) -> str:
    # Stop when length criteria is met or after 3 attempts
    if state.get("is_valid", False) or state.get("revision_count", 0) >= 3:
        return END
    return "generate_answer"


# Build the StateGraph
graph = StateGraph(LLMState)

graph.add_node("generate_answer", generate_answer)
graph.add_node("validate_length", validate_length)

graph.add_edge(START, "generate_answer")
graph.add_edge("generate_answer", "validate_length")
graph.add_conditional_edges(
    "validate_length",
    should_continue,
    {
        "generate_answer": "generate_answer",
        END: END,
    },
)

# Compile into a runnable workflow
workflow = graph.compile()

if __name__ == "__main__":
    initial_state: LLMState = {
        "question": "What is Crunchyroll?",
        "min_length": 40,
        "max_length": 90,
    }

    print(f"Question: {initial_state['question']}")
    print(
        f"Enforcing length: between {initial_state['min_length']} "
        f"and {initial_state['max_length']} words.\n"
    )

    final_state = workflow.invoke(cast(Any, initial_state))

    print("--- Workflow Summary ---")
    print(f"Word Count: {final_state.get('word_count')}")
    print(f"Iterations: {final_state.get('revision_count')}")
    print(f"Valid: {final_state.get('is_valid')}")
    print("\n--- Dedicated Answer ---")
    print(final_state["answer"])
