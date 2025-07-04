# 📊 Chat with Pandas
**An interactive Streamlit app for chatting with your pandas DataFrames using a LLM.**

---
## 🚀 Features

- **💬 Natural language queries**: Ask questions like "What's the average value in this column?" or "Show me the top 5 entries for Bob."
- **🔐 Safe code execution**: Only use pre-defined Python tools or SQL via duckdb to transform the pandas dataframe
- **🛠️ Extensible setup**: Easily swap in new LLM backends or add data visualization support.

---

## 📦 Installation

```bash
# Clone the repository and start with uv
git clone https://github.com/kzk2000/chat-with-pandas.git
cd chat-with-pandas
uv sync                         # creates ./.venv
export GOOGLE_API_KEY=....      # https://aistudio.google.com/ -> "Get API Key"  (it's free!)
uv run streamlit run app.py

# Alternatively: Install requirements via pip into your venv
pip install -e .                
streamlit run app.py
```
Then open your browser to: http://localhost:8501

## 📁 Project Structure
```
.
├── app.py                          # Streamlit app
├── LICENSE
├── pyproject.toml
├── README.md
├── src
│   ├── chat_with_pandas
│   │   ├── __init__.py
│   │   ├── tools.py                # Python functions registry
│   │   ├── types.py                # Pydantic classes that select from the tools.py
```


