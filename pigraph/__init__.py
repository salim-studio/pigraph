"""pigraph — fast, LangGraph-compatible agent graphs (+ data, ML/DL, RAG).

Quickstart:
    from pigraph import StateGraph, START, END, MemorySaver
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
    g.add_node("b", lambda s: {"y": s["x"] * 2})
    g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)
    app = g.compile(checkpointer=MemorySaver())
    print(app.invoke({"x": 0}))

Why fast:
- Plain-dict state, no pydantic, no deepcopy except at checkpoints
- Linear fast-path (no thread pool, no routing copies)
- Static edges resolved at compile time; parallel fan-out only on branches

Modules:
- pigraph.db: unified Database (sqlite/duckdb/sqlalchemy) + JsonFileSaver
- pigraph.data: ETL, cleaning, normalization, DataPipeline
- pigraph.ml: metrics, baseline models, make_ml_graph (sklearn optional)
- pigraph.dl: dependency-free SimpleMLP + Numericalizer + torch adapters
- pigraph.vector: lightweight InMemoryVectorStore for RAG
- pigraph.viz: mermaid/ascii rendering, plots, TimerCallback
- pigraph.utils: retry/cached/timed/parallel_map
"""
from __future__ import annotations

__version__ = "0.3.0"

from .types import END, START, Command, GraphInterrupt, Interrupt, Send, interrupt
from .graph import (CompiledGraph, MessageGraph, MessagesState, StateGraph,
                    StateSnapshot, add_messages)
from .checkpoint import (BaseCheckpointer, Checkpoint, InMemorySaver,
                         MemorySaver, MemorySaverCheckpoint, SqliteSaver)
from .prebuilt import ToolExecutor, ToolNode, create_react_agent, tools_condition
from .functional import entrypoint, task
from .db import Database, JsonFileSaver, csv_loader_node, sql_node
from .data import (DataPipeline, clean_missing, describe, etl_node,
                   fill_missing, group_by_agg, label_encode, load_csv,
                   load_json, normalize, one_hot_encode, save_csv, save_json,
                   standardize, train_test_split)
from .ml import (MajorityClassifier, MeanRegressor, accuracy, cross_validate,
                 f1, mae, make_eval_node, make_ml_graph, make_predict_node,
                 make_train_node, mse, precision, r2, recall, rmse)
from .dl import Numericalizer, SimpleMLP, chunk_text, make_dl_node
from .vector import Document, InMemoryVectorStore
from .viz import (TimerCallback, graph_to_ascii, graph_to_mermaid, plot_history,
                  save_mermaid)
from .utils import batch_apply, cached, parallel_map, retry, timed

__all__ = ["StateGraph", "MessageGraph", "CompiledGraph", "StateSnapshot", "MessagesState",
           "Command", "Send", "Interrupt", "GraphInterrupt", "interrupt",
           "START", "END", "add_messages",
           "MemorySaver", "InMemorySaver", "MemorySaverCheckpoint", "SqliteSaver",
           "BaseCheckpointer", "Checkpoint",
           "ToolNode", "ToolExecutor", "tools_condition", "create_react_agent",
           "entrypoint", "task",
           # db
           "Database", "JsonFileSaver", "sql_node", "csv_loader_node",
           # data
           "DataPipeline", "etl_node", "load_csv", "save_csv", "load_json", "save_json",
           "clean_missing", "fill_missing", "describe", "normalize", "standardize",
           "one_hot_encode", "label_encode", "train_test_split", "group_by_agg",
           # ml/dl
           "accuracy", "precision", "recall", "f1", "mse", "rmse", "mae", "r2",
           "MajorityClassifier", "MeanRegressor", "make_train_node", "make_predict_node",
           "make_eval_node", "make_ml_graph", "cross_validate",
           "SimpleMLP", "Numericalizer", "chunk_text", "make_dl_node",
           # vector/viz/utils
           "Document", "InMemoryVectorStore",
           "graph_to_mermaid", "graph_to_ascii", "save_mermaid", "plot_history",
           "TimerCallback", "retry", "cached", "timed", "parallel_map", "batch_apply"]
