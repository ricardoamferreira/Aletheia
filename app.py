import json
from typing import Annotated, List, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Load Keys
load_dotenv()

# UI Setup
st.set_page_config(page_title="Aletheia", page_icon="🏛️", layout="wide")
st.title("🏛️ Aletheia: Autonomous Truth Engine")


# Agent setup
class Competitor(BaseModel):
    name: str = Field(description="Name of the competitor")
    pricing: str = Field(description="Pricing model or specific cost if found")
    pros: str = Field(description="Key strengths")
    cons: str = Field(description="Key weaknesses")


class ResearchReport(BaseModel):
    summary: str = Field(description="Executive summary of the market landscape")
    competitors: List[Competitor] = Field(description="List of analysed competitors")
    sources: List[str] = Field(description="List of URLs used for research")


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    final_report: dict


search_tool = TavilySearch(max_results=3)


@tool
def submit_report(report: ResearchReport):
    """Call this tool when you have gathered all necessary information to submit the final report."""
    return "Report submitted."


tools = [search_tool, submit_report]

# Brain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tool_node(state: AgentState):
    last_msg = state["messages"][-1]
    if last_msg.tool_calls[0]["name"] == "submit_report":
        return {"final_report": last_msg.tool_calls[0]["args"]}

    # Otherwise, run standard tools (Search)
    return ToolNode(tools).invoke(state)


def should_continue(state: AgentState):
    last_msg = state["messages"][-1]

    # If no tool called, force it to stop (or loop back, but usually implies error here)
    if not last_msg.tool_calls:
        return END

    # If the tool called is "submit_report", we are done!
    if last_msg.tool_calls[0]["name"] == "submit_report":
        return END

    # Otherwise, go to tools to execute the search
    return "tools"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")  # Loop back after searching

agent_executor = builder.compile()


# 9. The UI
user_query = st.text_input(
    "Mission Objective:",
    placeholder="e.g. Find 3 competitors to Audible and compare pricing.",
)

if st.button("Initialize Agent") and user_query:
    st.info("Aletheia is active. Reasoning & Searching...")

    log_container = st.expander("Show Thinking Process", expanded=True)
    report_container = st.container()

    system_prompt = SystemMessage(
        content="""
        You are a senior research analyst. 
        Your goal is to gather data and submit a detailed report.
        
        CRITICAL RULES:
        1. You must NOT reply with conversational text or summaries.
        2. You must continue using the 'tavily_search_results_json' tool until you have all the data.
        3. When you call 'submit_report', you MUST pass the 'summary', 'competitors', and 'sources' fields. 
        4. DO NOT call 'submit_report' with empty arguments.
    """
    )

    # 2. Run the Agent
    events = agent_executor.stream(
        {
            "messages": [system_prompt, HumanMessage(content=user_query)],
            "final_report": {},
        },
        stream_mode="values",
    )

    final_data = None

    for event in events:
        if "messages" in event:
            last_msg = event["messages"][-1]

            # A. Visualise the Tools
            with log_container:
                if last_msg.type == "tool":
                    st.markdown(f"**🔎 Source Found:**")
                    st.code(last_msg.content[:300] + "...")

                # B. Check if the Agent is trying to submit the report
                elif last_msg.type == "ai" and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    tool_args = last_msg.tool_calls[0]["args"]

                    st.write(f"Thinking: calling {tool_name}...")

                    if tool_name == "submit_report":
                        st.warning(f"DEBUG: Raw Arguments received: {tool_args}")

                        if "report" in tool_args:
                            final_data = tool_args["report"]
                        else:
                            final_data = tool_args
    # 3. Render the Final Report
    if final_data:
        st.success("Report Generated Successfully!")

        with report_container:
            st.markdown(f"### 📝 {final_data.get('summary', 'Executive Summary')}")

            st.markdown("#### Competitor Analysis")
            # Convert the list of competitors to a nice dataframe
            if "competitors" in final_data:
                st.dataframe(final_data["competitors"])

            st.markdown("#### Sources")
            if "sources" in final_data:
                for s in final_data["sources"]:
                    st.markdown(f"- {s}")
    else:
        st.warning("Agent finished but did not submit a structured report.")

with st.sidebar:
    st.markdown("---")
    st.header("🧠 Neural Blueprint")

    try:
        graph_image = agent_executor.get_graph().draw_mermaid_png()
        st.image(graph_image, caption="Aletheia's Logic Flow")
    except Exception as e:
        st.warning("Visualisation requires extra dependencies.")
        st.info(f"Structure: {agent_executor.get_graph().draw_ascii()}")
