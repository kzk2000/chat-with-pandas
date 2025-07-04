# 📊 Chat with Pandas

**An interactive Streamlit chat for chatting with your pandas DataFrames using natural language and LLMs.**

---

## 🚀 Features

- **💬 Natural language queries**: Ask questions like "What's the average value in this column?" or "Show me the top 5 entries for Bob."
- **📈 Auto-generated code**: Behind the scenes, the tool crafts pandas code based on your input.
- **🔐 Safe code execution**: Only use pre-defined Python tools or SQL via duckdb
- **🛠️ Extensible setup**: Easily swap in new LLM backends or add data visualization support.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/kzk2000/chat-with-pandas.git
cd chat-with-pandas
uv sync
export GOOGLE_API_KEY=....      # https://aistudio.google.com/ -> "Get API Key"  (it's free!)

stream

# Install requirements
pip install -r requirements.txt
