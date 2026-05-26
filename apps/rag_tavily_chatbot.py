import streamlit as st
from rag_tavily_agent_executor import build_agent_executor

st.set_page_config(page_title="Drug Review RAG Agent", layout="wide")
st.title("💊 Drug Review RAG Agent")
st.caption("Ask questions about the local drug reviews dataset and enrich answers with Tavily web search.")

@st.cache_resource
def get_agent_executor():
    return build_agent_executor()


def extract_tavily_sources(steps):
    """
    Look for TavilySafeSearch observations and extract the Sources section.
    """
    sources = []

    for step in steps:
        if not isinstance(step, (tuple, list)) or len(step) != 2:
            continue

        action, observation = step
        tool_name = getattr(action, "tool", "")

        if tool_name != "TavilySafeSearch":
            continue

        observation_text = str(observation)

        if "Sources:" in observation_text:
            source_block = observation_text.split("Sources:", 1)[1].strip()
            if source_block:
                for line in source_block.splitlines():
                    line = line.strip()
                    if line and line not in sources:
                        sources.append(line)

    return sources

agent_executor = get_agent_executor()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Settings")
    show_steps = st.checkbox("Show intermediate steps", value=False)

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "steps" in message and show_steps:
            with st.expander("Intermediate steps"):
                st.write(message["steps"])

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
                result = agent_executor.invoke({"input": user_query})
                answer = result["output"]
                steps = result.get("intermediate_steps", [])

                sources = extract_tavily_sources(steps)

                if sources:
                    answer = answer + "\n\n### Sources\n" + "\n".join([f"- {s}" for s in sources])

                st.markdown(answer)

                if show_steps:
                    with st.expander("Intermediate steps"):
                        st.write(steps)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "steps": steps
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })