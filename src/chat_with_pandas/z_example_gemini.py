import json
import os
from typing import Optional, Dict

import duckdb
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

# --- MODEL DEFINITIONS ---

# Model 1: The simple, JSON-serializable "receipt" that the tool returns.
# The LLM will see this model's schema and its data.
class SQLToolOutput(BaseModel):
    """The result of a SQL tool execution."""
    success: bool
    sql: str
    message: str

# Model 2: The final, rich output that the developer wants.
# The LLM never sees this model's schema. It is for the final output only.
class AgentFinalOutput(BaseModel):
    """The final structured output from the agent's analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: str
    description: str
    success: bool
    sql: Optional[str] = None

    # This field is for Python use only and is excluded from any serialization.
    df: Optional[pd.DataFrame] = Field(default=None, exclude=True)


# --- Helper and Setup ---

def create_dataframe_info(df: pd.DataFrame) -> Dict:
    """Create comprehensive DataFrame metadata as a dictionary"""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_data": df.head(3).to_string(),
    }


def mask_sensitive_value(value: str) -> str:
    """Mask sensitive values while preserving format information"""
    if pd.isna(value):
        return value

    # Preserve structure/format while masking content
    if len(value) <= 2:
        return '*' * len(value)
    else:
        return value[0] + '*' * (len(value) - 1)


df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'JJ'],
    'age': [25, 30, 35, 28, 32],
    'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
    'salary': [50000, 75000, 80000, 70000, 55000]
})
df_info = create_dataframe_info(df)

df['name'].map(mask_sensitive_value)

system_prompt = f"""
You are an expert data analyst. Your goal is to write a single, correct DuckDB SQL query to answer the user's question and execute it using the `smart_sql` tool. The user's DataFrame is available as a table named 'df'.

Based on the result from the tool, you will then formulate the final answer.

DATASET OVERVIEW:
{json.dumps(df_info, indent=2)}
"""

# --- Agent and Tool Definition ---

# The agent's output_type is our desired rich model.
agent = Agent(
    model=GoogleModel(
        model_name='gemini-1.5-flash',
        provider=GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"]),
    ),
    model_settings={'temperature': 0.0, 'seed': 42},
    deps_type=pd.DataFrame,
    output_type=AgentFinalOutput, # <-- Use the rich, final model here
    system_prompt=system_prompt
)

# The tool's return type is the simple, serializable model.
@agent.tool
def smart_sql(ctx: RunContext[pd.DataFrame], sql: str) -> SQLToolOutput: # <-- Note the return type
    """Executes a DuckDB SQL query against the user's DataFrame."""
    try:
        input_df = ctx.deps
        with duckdb.connect() as conn:
            conn.register('df', input_df)
            result_df = conn.execute(sql).fetchdf()

        # The bridge: Store the complex object in the context state.
        ctx.state['result_df'] = result_df
        print(f"INFO: Tool executed. DataFrame of shape {result_df.shape} stored in context state.")

        # Return the simple "receipt" to the LLM.
        return SQLToolOutput(
            success=True,
            sql=sql,
            message=f"Successfully executed query. Result has {result_df.shape[0]} rows."
        )
    except Exception as e:
        return SQLToolOutput(
            success=False,
            sql=sql,
            message=f"SQL execution failed: {str(e)}"
        )

# --- Main Execution and Result Assembly ---

print("Running agent to get all IT department employees...")
result = agent.run_sync("Get all employees from the IT department", deps=df)

# The result.output is an instance of AgentFinalOutput, created by the LLM.
# Its `df` attribute is currently None.
final_output = result.output

print("\n--- LLM Final Summary (from AgentFinalOutput) ---")
print(final_output.model_dump_json(indent=2))

# Now, we manually assemble the final object.
# We retrieve the DataFrame from the context state...
retrieved_df = result.context.state.get('result_df')

# ...and assign it to our final output model's 'df' field.
if retrieved_df is not None:
    final_output.df = retrieved_df
    print("\nINFO: DataFrame successfully attached to the final output model.")

# --- Verification ---
print("\n--- Final Result with DataFrame ---")
if final_output.df is not None:
    print("DataFrame stored in `final_output.df`:")
    print(final_output.df)
    assert final_output.df.shape[0] == 2
    assert all(final_output.df['department'] == 'IT')
else:
    print("No DataFrame was generated or an error occurred.")