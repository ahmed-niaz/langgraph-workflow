import os
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict, cast

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables (.env searched upward automatically)
load_dotenv(find_dotenv())

# Minimize hallucination: temperature = 0 makes the output deterministic


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


# Initialize LLM and embeddings instances
base_llm = get_llm()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(
    base_dir, "Bangladesh _Stock_Market_Performance_2026.pdf"
)

# Safety measure for debugging purposes
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

# Checks if the PDF is there and loads pages
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e!s}")
    raise

# Chunking process
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)

# Apply splitter to loaded pages
pages_split = text_splitter.split_documents(pages)

persist_directory = r"E:\ai\2026\langgraph-workflow\workflow"
collection_name = "stock_market"

# Create persist directory if it does not exist
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Create the Chroma vector database
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print("Created the ChromaDB vector store")
except Exception as e:
    print(f"Error setting up ChromaDB: {e!s}")
    raise

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},  # K is the amount of chunks to return
)


@tool
def retriever_tool(query: str) -> str:
    """Search and return information about 2026 stock market performance.

    Args:
        query (str): Search query about stock market performance.

    Returns:
        str: Relevant document excerpts.
    """
    docs = retriever.invoke(query)

    if not docs:
        return (
            "I found no relevant documents on "
            "stock market performance in 2026."
        )

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]

# Bind tools with LLM
llm = base_llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState) -> bool:
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    tool_calls = getattr(result, "tool_calls", None)
    return bool(tool_calls)


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market
Performance in 2026 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market
performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow-up question,
you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}


# LLM Agent
def call_llm(state: AgentState) -> dict[str, Any]:
    """Call the LLM with the current state."""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages": [message]}


# Retriever Agent
def take_action(state: AgentState) -> dict[str, Any]:
    """Execute tool calls from the LLM's response."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])
    results = []
    for t in tool_calls:
        query = t["args"].get("query", "No query provided")
        print(f"Calling Tool: {t['name']} with query: {query}")

        if t["name"] not in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result: Any = (
                "Incorrect Tool Name, Please Retry and Select a tool "
                "from the List of Available tools."
            )
        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        results.append(
            ToolMessage(
                tool_call_id=t["id"],
                name=t["name"],
                content=str(result),
            )
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()


def _extract_text(content: Any) -> str:
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text", ""))
        return str(first)
    return str(content)


def running_agent() -> None:
    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\nWhat is your question: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        messages: list[BaseMessage] = [HumanMessage(content=user_input)]
        result = rag_agent.invoke(cast(Any, {"messages": messages}))

        print("\n=== ANSWER ===")
        print(_extract_text(result["messages"][-1].content))


if __name__ == "__main__":
    running_agent()
