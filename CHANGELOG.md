# Changelog

All notable changes to `pigraph` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3.0] - 2026-09-14
### Added
- Renamed package `pygraph` → `pigraph` (with backward-compatible `pygraph` alias).
- New modules: `pigraph.db` (unified Database + `JsonFileSaver`), `pigraph.data`
  (ETL/DataPipeline), `pigraph.ml` (metrics, baselines, `make_ml_graph`),
  `pigraph.dl` (dependency-free `SimpleMLP`), `pigraph.vector`
  (`InMemoryVectorStore` for RAG), `pigraph.viz`, `pigraph.utils`.
- English README, MIT license, logo/banner assets, CI workflow.

## [0.2.0] - 2026-09-11
### Added
- LangGraph-compatible `StateGraph`/`MessageGraph`, `Command`/`Send`,
  `interrupt`, checkpointing, `ToolNode`, `create_react_agent`, functional API.
