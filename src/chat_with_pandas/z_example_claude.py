import nest_asyncio
nest_asyncio.apply()

import pandas as pd
import duckdb
from typing import List, Dict, Any, Optional
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, ConfigDict
import json
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
import os

class DataFrameInfo(BaseModel):
    """Metadata about the DataFrame"""
    shape: tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    sample_data: str
    memory_usage: str
    null_counts: Dict[str, int]


class QueryResult(BaseModel):
    """Result of a DataFrame operation"""
    #model_config = ConfigDict(arbitrary_types_allowed=True)

    operation: str
    description: str
    result_shape: tuple[int, int]
    preview: str
    success: bool
    sql: Optional[str] = None
    #df: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None


class ConversationMemory(BaseModel):
    """Memory of previous operations"""
    operations: List[str] = []
    last_result_info: Optional[str] = None
    context_summary: str = ""


class DataFrameContext(BaseModel):
    """Rich context about the DataFrame"""
    info: DataFrameInfo
    memory: ConversationMemory = ConversationMemory()


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


# Enhanced agent with better context
def create_dataframe_agent(df: pd.DataFrame) -> Agent:
    """Create a context-aware DataFrame agent"""

    df_info = create_dataframe_info(df)
    context = DataFrameContext(info=df_info)

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

    agent = Agent(
        model=GoogleModel(
            model_name='gemini-2.5-flash',
            provider=GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"]),
        ),
        model_settings={'temperature': 0.0, 'seed': 42},
        deps_type=tuple[pd.DataFrame, DataFrameContext],
        output_type=QueryResult,
        system_prompt=system_prompt
    )

    agent.run_sync('get me the last row')
    #
    # @agent.tool
    # def get_dataframe_info(ctx: RunContext[tuple[pd.DataFrame, DataFrameContext]]) -> str:
    #     """Get comprehensive information about the DataFrame"""
    #     df, context = ctx.deps
    #     return f"""
    #     DataFrame Summary:
    #     - Shape: {df.shape}
    #     - Columns: {list(df.columns)}
    #     - Data Types: {df.dtypes.to_dict()}
    #     - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB
    #     - Missing Values: {df.isnull().sum().to_dict()}
    #
    #     Sample Data:
    #     {df.head().to_string()}
    #
    #     Recent Operations: {context.memory.operations[-3:] if context.memory.operations else 'None'}
    #     """

    @agent.tool
    def smart_filter(ctx: RunContext[tuple[pd.DataFrame, DataFrameContext]],
                     condition: str, description: str = "") -> QueryResult:
        """Filter DataFrame with intelligent error handling"""
        df, context = ctx.deps

        try:
            # Validate column names in condition
            for col in df.columns:
                if col in condition and col not in df.columns:
                    return QueryResult(
                        operation="filter",
                        description=f"Failed: Column '{col}' not found",
                        result_shape=(0, 0),
                        preview="",
                        success=False,
                        error_message=f"Column '{col}' does not exist"
                    )

            result_df = df.query(condition)

            # Update memory
            context.memory.operations.append(f"Filter: {condition}")
            context.memory.last_result_info = f"Filtered to {result_df.shape[0]} rows"

            return QueryResult(
                operation="filter",
                description=description or f"Filtered data using condition: {condition}",
                result_shape=result_df.shape,
                preview=result_df.head().to_string() if not result_df.empty else "No results",
                success=True
            )

        except Exception as e:
            return QueryResult(
                operation="filter",
                description=f"Filter failed: {str(e)}",
                result_shape=(0, 0),
                preview="",
                success=False,
                error_message=str(e)
            )

    @agent.tool
    def smart_sql(ctx: RunContext[tuple[pd.DataFrame, DataFrameContext]],
                  sql: str, description: str = "") -> QueryResult:
        """Execute SQL with better error handling and context"""
        df, context = ctx.deps

        try:
            conn = duckdb.connect()
            conn.register('df', df)

            result_df = conn.execute(sql).fetchdf()
            conn.close()

            # Update memory
            context.memory.operations.append(f"SQL: {sql[:50]}...")
            context.memory.last_result_info = f"SQL returned {result_df.shape[0]} rows"

            return QueryResult(
                operation="sql",
                description=description or f"Executed SQL query",
                result_shape=result_df.shape,
                preview=result_df.head().to_string() if not result_df.empty else "No results",
                success=True,
                sql =sql,
            )

        except Exception as e:
            return QueryResult(
                operation="sql",
                description=f"SQL execution failed: {str(e)}",
                result_shape=(0, 0),
                preview="",
                success=False,
                error_message=str(e)
            )

    kk = agent.run_sync("average age by name", deps=(df, context))
    kk.output.sql

    @agent.tool
    def analyze_column(ctx: RunContext[tuple[pd.DataFrame, DataFrameContext]],
                       column_name: str) -> QueryResult:
        """Analyze a specific column with statistics"""
        df, context = ctx.deps

        if column_name not in df.columns:
            return QueryResult(
                operation="analyze",
                description=f"Column '{column_name}' not found",
                result_shape=(0, 0),
                preview="",
                success=False,
                error_message=f"Column '{column_name}' does not exist"
            )

        col_data = df[column_name]

        if col_data.dtype in ['int64', 'float64']:
            stats = col_data.describe()
            analysis = f"""
            Numerical Analysis for '{column_name}':
            {stats.to_string()}

            Missing values: {col_data.isnull().sum()}
            Unique values: {col_data.nunique()}
            """
        else:
            analysis = f"""
            Categorical Analysis for '{column_name}':
            Value counts:
            {col_data.value_counts().head(10).to_string()}

            Missing values: {col_data.isnull().sum()}
            Unique values: {col_data.nunique()}
            """

        context.memory.operations.append(f"Analyzed column: {column_name}")

        return QueryResult(
            operation="analyze",
            description=f"Statistical analysis of column '{column_name}'",
            result_shape=df.shape,
            preview=analysis,
            success=True
        )

    @agent.tool
    def suggest_operations(ctx: RunContext[tuple[pd.DataFrame, DataFrameContext]]) -> str:
        """Suggest relevant operations based on the data"""
        df, context = ctx.deps

        suggestions = []

        # Data quality suggestions
        if df.isnull().sum().sum() > 0:
            suggestions.append("- Handle missing values with fillna() or dropna()")

        # Column type suggestions
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].nunique() < 10:
                    suggestions.append(f"- Analyze categorical column '{col}' with value_counts()")
            elif df[col].dtype in ['int64', 'float64']:
                suggestions.append(f"- Get statistics for numerical column '{col}'")

        # Size-based suggestions
        if df.shape[0] > 10000:
            suggestions.append("- Consider sampling for initial exploration")

        return f"""
        Suggested operations for your dataset:
        {chr(10).join(suggestions)}

        Recent operations: {context.memory.operations[-3:] if context.memory.operations else 'None'}
        """

    return agent


# Usage example
def chat_with_dataframe(df: pd.DataFrame, query: str) -> QueryResult:
    """Main function to chat with DataFrame"""
    agent = create_dataframe_agent(df)
    context = DataFrameContext(info=create_dataframe_info(df))

    result = agent.run_sync(query, deps=(df, context))
    return result.data


# Example usage
if __name__ == "__main__":

    # Sample DataFrame
    sample_df = df =  pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'department': ['HR', 'IT', 'Finance', 'IT', 'HR'],
        'salary': [50000, 75000, 80000, 70000, 55000],
        'years_experience': [2, 5, 8, 3, 4]
    })



    queries = [
        #"What does this dataset contain?",
        "Show me employees in the IT department. Use SQL to answer",
        # "What's the average salary by department?",
        # "Find employees with more than 4 years experience",
        # "Analyze the age distribution"
    ]

    for query in queries:
        break
        print(f"\n🤖 Query: {query}")
        result = chat_with_dataframe(sample_df, query)
        print(f"✅ {result.description}")
        print(f"📊 Result shape: {result.result_shape}")
        print(f"👀 Preview:\n{result.preview}")
        if not result.success:
            print(f"❌ Error: {result.error_message}")

