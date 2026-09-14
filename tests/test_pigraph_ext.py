"""اختبارات التوسعة 0.3.0: db / data / ml / dl / vector / viz / utils."""
from pigraph import (END, START, Database, DataPipeline, InMemoryVectorStore,
                     JsonFileSaver, MajorityClassifier, SimpleMLP, StateGraph,
                     TimerCallback, accuracy, clean_missing, cross_validate,
                     describe, f1, graph_to_ascii, graph_to_mermaid, group_by_agg,
                     make_dl_node, make_ml_graph, mse, normalize, one_hot_encode,
                     retry, train_test_split)


def test_db_crud_and_nodes():
    db = Database(":memory:")
    db.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    assert db.executemany("INSERT INTO t VALUES (?,?)", [(1, "a"), (2, "b")]) == 2
    rows = db.query("SELECT * FROM t ORDER BY id")
    assert [r["name"] for r in rows] == ["a", "b"]
    assert "t" in db.tables()
    g = StateGraph(dict)
    g.add_node("load", db.read_node("SELECT * FROM t", out_key="rows"))
    g.add_node("write", db.write_node("t2", in_key="rows"))
    g.add_edge(START, "load"); g.add_edge("load", "write"); g.add_edge("write", END)
    out = g.compile().invoke({})
    assert out["inserted"] == 2
    assert len(db.query("SELECT * FROM t2")) == 2
    db.close()


def test_db_csv_and_json_saver(tmp_path):
    import csv as _csv
    p = tmp_path / "a.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=["x", "y"])
        w.writeheader(); w.writerows([{"x": "1", "y": "a"}, {"x": "2", "y": "b"}])
    db = Database(":memory:")
    rows = db.load_csv(str(p), table="csv_t")
    assert len(rows) == 2 and len(db.query("SELECT * FROM csv_t")) == 2
    sv = JsonFileSaver(str(tmp_path / "c.json"))
    sv.put("t1", {"x": 1}, 1)
    assert sv.get("t1") == {"x": 1}
    assert len(sv.history("t1")) == 1
    db.close()


def test_data_pipeline():
    rows = [{"a": "1", "c": "x"}, {"a": "", "c": "y"}, {"a": "3", "c": "x"}]
    clean = clean_missing(rows, strategy="mean")
    assert all(r["a"] != "" for r in clean)
    norm = normalize([{"a": 0}, {"a": 10}], columns=["a"])
    assert norm[0]["a"] == 0.0 and norm[1]["a"] == 1.0
    enc = one_hot_encode([{"c": "x"}, {"c": "y"}], "c")
    assert enc[0]["c_x"] == 1 and enc[1]["c_y"] == 1
    d = describe([{"a": 1}, {"a": 3}])
    assert d["columns"]["a"]["mean"] == 2.0
    assert group_by_agg([{"r": "a", "v": 2}, {"r": "a", "v": 4}], "r", "mean", "v")[0]["mean"] == 3.0
    tr, te = train_test_split([{"i": i} for i in range(10)], test_size=0.2, seed=1)
    assert len(tr) == 8 and len(te) == 2
    pipe = DataPipeline([lambda r: clean_missing(r, strategy="mean")])
    app = pipe.to_graph(in_key="rows", out_key="dataset")
    out = app.invoke({"rows": rows})
    assert "dataset" in out


def test_ml_graph_and_metrics():
    assert accuracy([0, 1, 1], [0, 0, 1]) == 2 / 3
    assert mse([1, 2], [1, 4]) == 2.0
    assert 0 <= f1([0, 1, 1], [0, 0, 1]) <= 1
    ds = [{"x": 1, "label": 0}, {"x": 2, "label": 0}, {"x": 9, "label": 1}, {"x": 8, "label": 1}]
    app = make_ml_graph(target="label", metrics=("accuracy",))
    out = app.invoke({"dataset": ds})
    assert "model" in out and "preds" in out and "metrics" in out
    assert out["metrics"]["accuracy"] >= 0.5
    cv = cross_validate(MajorityClassifier, ds, target="label", k=2)
    assert "mean" in cv and len(cv["scores"]) == 2


def test_dl_mlp_and_node():
    mlp = SimpleMLP(input_dim=1, hidden=[4], output_dim=1, task="regression")
    mlp.fit([[0], [1], [2]], [0, 2, 4], epochs=30)
    preds = mlp.predict([[3]])
    assert isinstance(preds, list) and len(preds) == 1
    g = StateGraph(dict)
    g.add_node("dl", make_dl_node(target="label", task="binary", epochs=5))
    g.add_edge(START, "dl"); g.add_edge("dl", END)
    out = g.compile().invoke({"dataset": [{"f": 0, "label": 0}, {"f": 5, "label": 1}] * 4})
    assert "dl_model" in out


def test_vector_store_and_rag_node():
    store = InMemoryVectorStore()
    store.add_texts(["الذكاء الاصطناعي", "قواعد البيانات SQL", "تعلم الآلة"])
    hits = store.search("ذكاء", k=2)
    assert hits and "text" in hits[0]
    g = StateGraph(dict)
    g.add_node("ret", store.retrieval_node(in_key="question", out_key="context", k=2))
    g.add_edge(START, "ret"); g.add_edge("ret", END)
    out = g.compile().invoke({"question": "ما هو الذكاء الاصطناعي؟"})
    assert "context" in out and len(out["context"]) > 0


def test_viz_and_utils():
    g = StateGraph(dict)
    g.add_node("a", lambda s: {"x": 1})
    g.add_edge(START, "a"); g.add_edge("a", END)
    app = g.compile()
    assert "a" in graph_to_mermaid(app) and "a" in graph_to_ascii(app)
    tc = TimerCallback()
    n = tc.timed(lambda s: {"x": 1}, "a")
    assert n({}) == {"x": 1} and tc.report()["a"]["n"] == 1
    calls = {"n": 0}

    @retry(tries=3)
    def flaky(state):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("boom")
        return {"ok": True}

    assert flaky({}) == {"ok": True}
