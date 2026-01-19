import json
from typing import Annotated, List, TypedDict

import requests
import streamlit as st
from bs4 import BeautifulSoup
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


@tool
def scrape_website(url: str):
    """Scrapes the content of a specific URL to get detailed information."""
    try:
        # 1. Fake a browser header
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)

        # 2. Parse the HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # 3. Kill javascript and styles (we only want text)
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()

        # 4. Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # 5. Limit to first 5,000 characters to save tokens
        return clean_text[:5000]

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


search_tool = TavilySearch(max_results=3)


@tool
def submit_report(summary: str, competitors: List[Competitor], sources: List[str]):
    """Call this tool when you have gathered all necessary information to submit the final report.

    Args:
        summary: Executive summary of the market landscape
        competitors: List of analysed competitors
        sources: List of URLs used for research
    """
    return "Report submitted."


tools = [search_tool, scrape_website, submit_report]

# Brain
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def tool_node(state: AgentState):
    """Executes tools. If submit_report is called, updates the final_report state."""
    last_msg = state["messages"][-1]

    # Check if we have tool *calls*
    if not last_msg.tool_calls:
        # Should not happen based on logic, but specific safety check
        return ToolNode(tools).invoke(state)

    final_report_data = None

    # Check all tool calls for submit_report
    for tool_call in last_msg.tool_calls:
        if tool_call["name"] == "submit_report":
            final_report_data = tool_call["args"]
            break  # Take the first one if multiple (unlikely)

    # Run all tools normally to generate valid ToolMessages for history
    result = ToolNode(tools).invoke(state)

    # If we captured a report, add it to the state update
    if final_report_data:
        result["final_report"] = final_report_data

    return result


def should_continue(state: AgentState):
    """Decide whether to continue to tools or end."""
    last_msg = state["messages"][-1]

    # If no tool called, we can't do anything -> END
    if not last_msg.tool_calls:
        return END

    # Ensure we define the logic for going to tools
    return "tools"


def route_after_tools(state: AgentState):
    """Check if we should end or go back to agent."""
    # If final_report is populated, we are done
    if state.get("final_report"):
        return END

    # Otherwise, loop back to the agent
    return "agent"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_conditional_edges("tools", route_after_tools, ["agent", END])

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
        1. Use 'tavily_search_results_json' to find relevant URLs.
        2. MANDATORY: Use 'scrape_website' to read the actual content of promising URLs. Do not rely solely on search snippets.
        3. DO NOT REPLY WITH TEXT. You must call the 'submit_report' tool to complete the mission.
        4. If you have enough info, call 'submit_report' immediately.
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
        # Check if the final report is in the state (most reliable)
        if "final_report" in event and event["final_report"]:
            final_data = event["final_report"]

        if "messages" in event:
            last_msg = event["messages"][-1]

            # A. Visualise the Tools
            with log_container:
                if last_msg.type == "tool":
                    st.markdown(f"**🔎 Source Found:**")
                    st.code(last_msg.content[:300] + "...")

                # B. Show thinking (optional, mostly for debug now since we capture final_report from state)
                elif last_msg.type == "ai" and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    tool_args = last_msg.tool_calls[0]["args"]

                    st.write(f"Thinking: calling {tool_name}...")

                    if tool_name == "submit_report":
                        st.info("Submitting report...")
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
