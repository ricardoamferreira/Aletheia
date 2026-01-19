# 🏛️ Aletheia: Autonomous Truth Engine

Aletheia is an autonomous AI research agent designed to perform deep market analysis, verify facts, and generate structured executive reports. Unlike standard chatbots, Aletheia uses a **cyclic graph architecture** (LangGraph) to iterate on searches, self-correct, and ensure data accuracy before submitting a final structured payload.

## 🚀 Key Features

* **Cyclic Reasoning Loop:** Built on `LangGraph`, the agent cycles through a "Reason → Act → Observe" loop until it has sufficient data.
* **Structured Data Extraction:** Enforces a rigid Pydantic schema to output clean JSON data (tables) rather than unstructured text.
* **Defensive State Management:** Handles nested tool outputs and partial failures robustly.
* **Live Thinking Process:** Visualises the agent's internal thought process and tool usage in real-time via Streamlit.

## 🛠️ Tech Stack

* **Core:** Python 3.10+
* **Orchestration:** LangChain & LangGraph (StateGraph implementation)
* **LLM:** OpenAI GPT-4o
* **Search Engine:** Tavily AI (Optimised for LLM retrieval)
* **Frontend:** Streamlit

## ⚙️ Architecture

The agent follows a custom StateGraph workflow:
1.  **Input:** User provides a mission objective (e.g., "Find competitors to Spotify").
2.  **Reasoning Node:** The LLM decides whether to search for more info or finish.
3.  **Tool Node:** Executes search queries using Tavily.
4.  **Loop:** The graph cycles back to the Reasoning Node with new evidence.
5.  **Termination:** Once data is sufficient, the agent calls the `submit_report` tool, triggering the final UI render.

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Aletheia.git](https://github.com/YOUR_USERNAME/Aletheia.git)
    cd Aletheia
    ```

2.  **Set up the Virtual Environment**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # Mac/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add your API keys:
    ```ini
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run app.py