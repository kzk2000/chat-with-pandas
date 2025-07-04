import json
import os
from typing import List, Dict, Optional

import duckdb
import pandas as pd
from pydantic import BaseModel, PrivateAttr
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


class DataFrameInfo(BaseModel):
    """Metadata about the DataFrame"""
    shape: tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    sample_data: str
    memory_usage: str
    null_counts: Dict[str, int]


class DataFrameOutput(BaseModel):
    operation: str = None
    description: str = None
    success: bool = None
    sql: Optional[str] = None
    error_message: Optional[str] = None
    df_set: bool = False
    _df: Optional[pd.DataFrame] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        if 'df' in data:
            self._df = data.get('df')
            self.df_set = True


def create_dataframe_info(df: pd.DataFrame) -> DataFrameInfo:
    """Create comprehensive DataFrame metadata"""
    return DataFrameInfo(
        shape=df.shape,
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        sample_data=df.head(3).to_string(),
        memory_usage=f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        null_counts={col: df[col].isnull().sum() for col in df.columns}
    )


# Sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'salary': [50000, 75000, 80000, 70000, 55000],
    'years_experience': [2, 5, 8, 3, 4]
})
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

Use the available tools to analyze and transform the data.
"""
print(system_prompt)

if False:
    data = dict(sql = "test", _df = pd.DataFrame(np.random.randn(5, 3), columns=list("abc")))
    kk = DataFrameOutput(**data)
    kk._df

agent = Agent(
    model=GoogleModel(
        model_name='gemini-2.5-flash',
        provider=GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"]),
    ),
    model_settings={'temperature': 0.0, 'seed': 42},
    deps_type=pd.DataFrame,
    output_type=DataFrameOutput,
    system_prompt=system_prompt
)


@agent.tool
def smart_sql(ctx: RunContext[pd.DataFrame], sql: str, description: str = "") -> DataFrameOutput:
    """Execute SQL with better error handling and context"""
    try:
        df = ctx.deps

        with duckdb.connect() as conn:
            conn.register('df', df)
            result_df = conn.execute(sql).fetchdf()

        tmp = DataFrameOutput(
            operation="sql",
            description=description or f"Executed SQL query",
            success=True,
            sql=sql,
            df=result_df,
        )
        print(10 * "*")
        print(f"{tmp._df=}")
        return tmp

    except Exception as e:
        return DataFrameOutput(
            operation="sql",
            description=f"SQL execution failed: {str(e)}",
            success=False,
            error_message=str(e)
        )


kk = agent.run_sync('get me the last 2 rows', deps=df)
print(kk.output.model_dump_json(indent=2))
kk.mo

if kk.success:
    final_df = kk.context.state.get('result_df')
# kk.output._df
kk.output.__private_attributes__
#kk.all_messages()
