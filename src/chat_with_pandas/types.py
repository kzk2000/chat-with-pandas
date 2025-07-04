from typing import Literal, List, Union, Dict

from pydantic import BaseModel, Field


class DataFrameInfo(BaseModel):
    """Metadata about the DataFrame"""
    shape: tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    sample_data: str
    memory_usage: str
    null_counts: Dict[str, int]


class FilterRows(BaseModel):
    function: Literal["filter_rows"]
    condition: str = Field(
        description="""filter condition to be used with pandas.DataFrame.query(). Always
convert string values to lowercase, e.g df.query('name == "charlie"') -> `df.query("name.str.lower() == @search_term.lower()) 
"""
    )


class SortRows(BaseModel):
    function: Literal["sort_rows"]
    by: str
    ascending: bool = True


class TopRows(BaseModel):
    function: Literal["top_rows"]
    n: int


class SelectColumns(BaseModel):
    function: Literal["select_columns"]
    columns: List[str]


class GroupBy(BaseModel):
    function: Literal["group_by"]
    group_by_cols: List[str]
    metric_cols: List[str]
    metric: Literal["first", "last", "mean", "median", "min", "max"] = Field(
        description="""group by metric used for pandas.DataFrame.groupby(group_by_cols)[metric_cols].agg(metric).
if unclear, use last."""
    )


# TODO: RunDuckDBSQL handles this well, and doesn't require a date index on the pandas data frame
# class Resample(BaseModel):
#     function: Literal["resample"]
#     freq: Literal["D", "W-MON", "M", "Q", "Y"] = Field(
#         description="""resamples a pandas dataframe via df.resample(freq).last()"""
#     )


class RunDuckDBSQL(BaseModel):
    function: Literal["run_duckdb_sql"]
    sql: str = Field(
        description="""uses duckdb to run a sql query on a dataframe"""
    )

AgentPlan = Union[FilterRows, SortRows, TopRows, SelectColumns, GroupBy, RunDuckDBSQL]
