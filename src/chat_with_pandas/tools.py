import json
import os

import duckdb
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from chat_with_pandas.types import AgentPlan, DataFrameInfo

SAFE_TOOLS = {}


def register_tool(fn):
    SAFE_TOOLS[fn.__name__] = fn
    return fn


@register_tool
def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    SAFE_TOOLS['filter_rows'] = filter_rows
    return df.query(condition)


@register_tool
def sort_rows(df: pd.DataFrame, by: str, ascending: bool = True) -> pd.DataFrame:
    return df.sort_values(by=by, ascending=ascending)


@register_tool
def top_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n)


def last_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.tail(n)


@register_tool
def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df[columns]


@register_tool
def group_by(df: pd.DataFrame, group_by_cols: list, metric_cols: list, metric) -> pd.DataFrame:
    return df.groupby(group_by_cols)[metric_cols].agg(metric)


@register_tool
def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return df.resample(freq).last()


@register_tool
def run_duckdb_sql(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    """Executes a SQL query against the in-memory dataframe using DuckDB syntax.
    Instructions:
    * Always use `ILIKE` for case-insensitive filtering of string columns
    * Always use `DATE_TRUNC` for resampling
    """
    with duckdb.connect() as conn:
        conn.register('df', df)
        result_df = conn.execute(sql).fetchdf()

    return result_df


def create_dataframe_info(df: pd.DataFrame) -> DataFrameInfo:
    """Create comprehensive DataFrame metadata"""
    return DataFrameInfo(
        shape=df.shape,
        columns=list(df.columns),
        dtypes={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        sample_data=df.head(3).to_string(),
        memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        null_counts={col: df[col].isnull().sum() for col in df.columns}
    )

def generate_system_prompt(df: pd.DataFrame = pd.DataFrame()):
    df_info = create_dataframe_info(df)

    # Dynamic system prompt with DataFrame context
    system_prompt = f"""
You're an expert data analyst assistant working with a DataFrame.

DATASET OVERVIEW:
- Shape: {df_info.shape[0]:,} rows × {df_info.shape[1]} columns
- Columns: {', '.join(df_info.columns)}
- Data Types: {json.dumps(df_info.dtypes, indent=2)}
- Memory Usage: {df_info.memory_usage}
- Missing Values: {json.dumps(df_info.null_counts, indent=2)}

SAMPLE DATA:
{df_info.sample_data}

INSTRUCTIONS:
1. Always validate column names exist before operations
2. Provide clear descriptions of what you're doing
3. Handle errors gracefully
4. Summarize results meaningfully
5. Remember previous operations in the conversation

You have access to various tools: If you can, always prefer to use of the `run_duckdb_sql` tool.
"""
    return system_prompt



agent = Agent(
    model=GoogleModel(
        #model_name='gemini-2.5-flash',
        model_name='gemini-2.5-pro',
        provider=GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"]),
    ),
    model_settings={'temperature': 0.0, 'seed': 42},
    output_type=[AgentPlan],
    system_prompt="""Generate a step by step plan as list."""  # we'll overwrite this in the app
)

if __name__ == '__main__':
    active_df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [88, 92, 85]
    })


    # agent._system_prompts
    # system_prompt = "You generate exactly one function call to modify the DataFrame."

    result = agent.run_sync("select top 2 rows, filter name for Bob, be case insensitive, use sql, format SQL nicely")
    print(result)
    # kwargs = result.output.__dict__
    # func_name = kwargs.pop('function')
    #
    # new_df = SAFE_TOOLS[func_name](active_df, **kwargs)
    # print(new_df)
