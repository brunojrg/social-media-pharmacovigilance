import os
from  dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import tiktoken

load_dotenv()

def build_agent_executor():
    # CSV vector database
    def get_vectorstore():
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        return Chroma(
            persist_directory="../drug-review-chroma_db",
            embedding_function=embeddings
        )

    # CSV search tool
    vs = get_vectorstore()

    def retrieve_csv(query: str, k: int = 5) -> str:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k)

        if not docs:
            return "No relevant records found in the local drug review database."

        results = []
        for i, doc in enumerate(docs, 1):
            row_id = doc.metadata.get("row", "unknown")
            text = doc.page_content[:300]
            results.append(f"[Document {i} | row={row_id}]\n{text}")

        return "\n\n".join(results)


    # Load GPT-4o-mini model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # Tokenizer for GPT-4
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Tavily instance
    tavily_search = TavilySearchResults()

    # Define safe search function (trim to 3500 tokens)
    def safe_search(query: str) -> str:
        """
        Performs a Tavily web search and ensures the result is short enough
        to fit within the GPT-4o-mini token limit.
        """
        result = tavily_search.invoke({"query": query})
        
        answer_text = ""
        source_lines = []
        body_parts = []

        # Handle result format
        if isinstance(result, dict):
            answer_text = result.get("answer", "")

            results_list = result.get("results", [])
            for i, item in enumerate(results_list, 1):
                title = item.get("title", "Untitled source")
                url = item.get("url", "")
                content = item.get("content", "")

                body_parts.append(f"[Source {i}] {content}")
                source_lines.append(f"{i}. {title} - {url}")

        elif isinstance(result, list):
            for i, item in enumerate(result, 1):
                if isinstance(item, dict):
                    title = item.get("title", "Untitled source")
                    url = item.get("url", "")
                    content = item.get("content", "")

                    body_parts.append(f"[Source {i}] {content}")
                    source_lines.append(f"{i}. {title} - {url}")
                else:
                    body_parts.append(str(item))

        else:
            body_parts.append(str(result))

        result_text = ""

        if answer_text:
            result_text += f"Answer: {answer_text}\n\n"

        if body_parts:
            result_text += "Evidence:\n" + "\n\n".join(body_parts) + "\n\n"

        if source_lines:
            result_text += "Sources:\n" + "\n".join(source_lines)
        
        # Count and trim tokens
        tokens = encoding.encode(result_text)
        trimmed = encoding.decode(tokens[:3500])  
        return trimmed

    # Define the tool
    tools = [
        Tool(
            name="DrugReviewRetriever",
            func=retrieve_csv,
            description=(
                "Searches the local Chroma database built from DrugReviews.csv. "
                "Use this for questions about patient reviews, drug effectiveness, "
                "side effects, satisfaction, or patterns present in the dataset." \
                "Use this for statistics of the reviews, such as the drugs with the most review, " \
                "or the most reported side effects for each drug"
            )
        ),
        Tool(
            name="TavilySafeSearch",
            func=safe_search,
            description=(
                "Web search tool for external medical or healthcare information. "
                "It trims output to avoid token overflow with GPT-4o-mini."
            )
        )
    ]

    def synthesize_from_steps(llm, query, intermediate_steps):
        observations = []

        for i, (action, observation) in enumerate(intermediate_steps, 1):
            tool_name = getattr(action, "tool", "unknown_tool")
            tool_input = getattr(action, "tool_input", "")

            observations.append(
                f"[Step {i}]\n"
                f"Tool: {tool_name}\n"
                f"Tool Input: {tool_input}\n"
                f"Observation:\n{str(observation)[:600]}"
            )

        joined_observations = "\n\n".join(observations).strip()

        if not joined_observations:
            return "I couldn’t gather enough tool output to produce a final answer."

    template = """
    You are a data scientist specialized in healthcare and pharmacovigilance.

    Previous conversation:
    {chat_history}

    You have access to the following tools:
    {tools}

    Use this format:

    Question: the input question you must answer
    Thought: think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat as needed)
    Thought: I now know the final answer
    Final Answer: the final answer to the original question

    Guidelines:
    - Prefer DrugReviewRetriever for dataset-specific questions. If a review has a rating ≤4 consider it a negative revies,
    if the rating is 5-6 consider it neutral, and if it's ≥7 consider it positive.
    - Use TavilySafeSearch for external medical knowledge, guidelines, or recent information.
    - For any general medical or pharmacology question, you must use TavilySafeSearch before giving the Final Answer.
    - When using TavilySafeSearch, include a "Sources" section in the final answer.
    - If TavilySafeSearch is used, include a short Sources section in the Final Answer.
    - Only include sources that came from the TavilySafeSearch tool output.
    - Use both tools when the user asks for dataset analysis plus medical explanation.
    - If information is uncertain or incomplete, say so clearly.
    - Be concise, accurate, and evidence-based.

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        tools=tools,
        llm=llm,
        agent=agent,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=180,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

    history_store = {}

    def get_session_history(session_id: str):
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        return history_store[session_id]

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return agent_with_history