import pandas as pd
import numpy as np
import streamlit as st

from chat_with_pandas.tools import agent, SAFE_TOOLS, generate_system_prompt

st.set_page_config(layout="wide")

if "active_df" not in st.session_state:
    n = 100
    st.session_state.df_original = pd.DataFrame(
        data={
            "ds": pd.date_range('2025-01-01', periods=n, freq='D'),
            "name": np.random.choice(["Alice", "Bob", "Charlie"], n),
            "age": np.random.randint(10, 15, n),
            "score": np.random.randint(80, 97, n)},
    )
    st.session_state.active_df = st.session_state.df_original.copy()

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📊 Data Chat")
# user_input = st.text_input("Instruction:", key='instruction')
user_input = st.chat_input(placeholder='Press \'r\' to reset')

if user_input in ('r', 'r '):
    st.session_state.active_df = st.session_state.df_original.copy()
    st.session_state.history = []
#  st.dataframe(st.session_state.active_df, hide_index=True)
elif user_input:
    agent._system_prompts = (generate_system_prompt(st.session_state.active_df, ))
    result = agent.run_sync(user_input)
    if hasattr(result, "output"):
        kwargs = result.output.__dict__
        st.write(result.__str__())
        st.write(kwargs.get('sql', ''))
        func_name = kwargs.pop('function')
        try:
            new_df = SAFE_TOOLS[func_name](st.session_state.active_df, **kwargs)
            st.session_state.history.append((user_input, new_df.copy(), result.__str__()))
            st.session_state.active_df = new_df
            st.success("✅ Applied transformation")
        except Exception as e:
            st.error(f"Could not request - do nothing!\nerror_message={str(e)}")

st.dataframe(st.session_state.active_df, hide_index=True)

st.sidebar.subheader("History")
if st.session_state.history:
    for idx, (instr, _, res) in enumerate(st.session_state.history[::-1], 1):
        st.sidebar.markdown(f"{idx}. **{instr}** → `{res}`")
