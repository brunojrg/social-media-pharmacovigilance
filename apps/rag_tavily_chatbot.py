import streamlit as st
import uuid
from rag_tavily_agent_executor import build_agent_executor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

st.set_page_config(page_title="Drug Review RAG Agent", layout="wide")
st.title("💊 Drug Review RAG Agent")
st.caption("Ask questions about the local drug reviews dataset and enrich answers with Tavily web search.\n\n"
"This application is provided strictly for educational, research, and exploratory data-science purposes and "
"does not provide medical advice, diagnosis, treatment recommendations, prescribing guidance, "
"or individualized clinical judgment and should not be used as a substitute for consultation with a qualified healthcare professional.")

@st.cache_resource
def get_agent_executor():
    return build_agent_executor()

agent_executor = get_agent_executor()

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Settings")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_intermediate_steps(steps):
    if not steps:
        return "No intermediate steps were returned."

    blocks = []
    for i, (action, observation) in enumerate(steps, 1):
        tool_name = getattr(action, "tool", "unknown_tool")
        tool_input = getattr(action, "tool_input", "")
        obs_text = str(observation)[:2000]

        blocks.append(
            f"### Step {i}\n"
            f"**Tool:** {tool_name}\n\n"
            f"**Input:** `{tool_input}`\n\n"
            f"**Observation:**\n{obs_text}"
        )

    return "\n\n---\n\n".join(blocks)


def synthesize_from_steps(llm, user_query, intermediate_steps):
    observations = []

    for i, (action, observation) in enumerate(intermediate_steps, 1):
        tool_name = getattr(action, "tool", "unknown_tool")
        tool_input = getattr(action, "tool_input", "")
        obs_text = str(observation)[:2000]

        observations.append(
            f"[Step {i}]\n"
            f"Tool: {tool_name}\n"
            f"Tool Input: {tool_input}\n"
            f"Observation:\n{obs_text}"
        )

    joined_observations = "\n\n".join(observations).strip()

    if not joined_observations:
        return "I couldn’t gather enough tool output to produce a final answer."

    prompt = f"""
You are a healthcare data assistant.
Write a concise final answer using only the tool observations below.
Do not invent facts. If the observations are incomplete, say so clearly.

User question:
{user_query}

Tool observations:
{joined_observations}

Final concise answer:
"""

    response = llm.invoke(prompt)
    return response.content

user_query = st.chat_input(
    "Ask about side effects, effectiveness, patient sentiment, or compare reviews with web information..."
)

if user_query:
    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing reviews and searching the web..."):
            try:
                result = agent_executor.invoke({"input": user_query},
                config={"configurable": {"session_id": st.session_state.chat_session_id}})
                answer = result["output"]
                steps = result.get("intermediate_steps", [])

                if "Agent stopped due to iteration limit or time limit" in answer and steps:
                    intermediate_text = format_intermediate_steps(steps)
                    fallback_answer = synthesize_from_steps(llm, user_query, steps)

                    answer = (
                        "The agent stopped before producing a normal final answer.\n\n"
                        "## Intermediate results\n\n"
                        f"{intermediate_text}\n\n"
                        "## Fallback answer\n\n"
                        f"{fallback_answer}"
                    )

            except Exception as e:
                answer = f"Error: {str(e)}"
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()