# pygraph — بديل langgraph الأسرع ⚡

مكتبة مطابقة لواجهة `langgraph` (نفس الأسماء: `StateGraph`, `START`, `END`, `Command`, `Send`, `interrupt`, `MemorySaver`...) لكن بتنفيذ أخف وأسرع: بدون pydantic، بدون نسخ عميق إلا عند checkpoint، مع fast-path للسلاسل الخطية وتوازٍ حقيقي عند التفرع.

## تنصيب

```bash
pip install -e .   # الحزمة تشمل pygraph ضمن pyproject.toml
```

## مثال سريع

```python
from pygraph import StateGraph, START, END, MemorySaver

g = StateGraph(dict)
g.add_node("a", lambda s: {"x": s.get("x", 0) + 1})
g.add_node("b", lambda s: {"y": s["x"] * 2})
g.add_edge(START, "a"); g.add_edge("a", "b"); g.add_edge("b", END)

app = g.compile(checkpointer=MemorySaver())
print(app.invoke({"x": 0}))  # {'x': 1, 'y': 2}
```

## مطابقة langgraph

| ميزة langgraph | pygraph |
|---|---|
| `StateGraph` / `MessageGraph` / `START` / `END` | ✅ |
| `add_messages` + `MessagesState` (دمج + upsert حسب `id` + `RemoveMessage`) | ✅ |
| `Command(update, goto)` + `Send(node, arg)` | ✅ |
| `interrupt()` / `Command(resume=...)` (human-in-the-loop) | ✅ |
| `interrupt_before` / `interrupt_after` | ✅ |
| `invoke` / `stream` / `batch` + `ainvoke` / `astream` / `abatch` | ✅ |
| `stream_mode`: `updates` / `values` / `messages` / `debug` | ✅ |
| `get_state` / `get_state_history` / `update_state` | ✅ |
| `MemorySaver` / `SqliteSaver` | ✅ |
| Subgraphs (عقدة = `CompiledGraph`) | ✅ |
| `ToolNode` / `tools_condition` / `create_react_agent` | ✅ (`pygraph.prebuilt`) |
| `@entrypoint` / `@task` | ✅ (`pygraph.functional`) |
| Reducers عبر `Annotated[list, fn]` | ✅ |

## أمثلة

### تفرع شرطي
```python
g = StateGraph(dict)
g.add_node("s", lambda s: {"n": 5})
g.add_node("even", lambda s: {"r": "even"})
g.add_node("odd", lambda s: {"r": "odd"})
g.set_entry_point("s")
g.add_conditional_edges("s", lambda s: "E" if s["n"] % 2 == 0 else "O",
                        {"E": "even", "O": "odd"})
```

### توازي Send (يُنفذ متوازياً عبر ThreadPool)
```python
from typing import Annotated
from typing import TypedDict
class S(TypedDict, total=False):
    vals: Annotated[list, lambda a, b: (a or []) + b]

g = StateGraph(S)
g.add_node("fan", lambda s: [Send("w", {"i": i}) for i in range(8)])
g.add_node("w", lambda s: {"vals": [s["i"]]})
g.set_entry_point("fan")
```

### Human-in-the-loop
```python
from pygraph import interrupt, Command
def node(s):
    ans = interrupt({"question": "هل أوافق؟"})
    return {"ans": ans}

app = g.compile(checkpointer=MemorySaver())
out = app.invoke({"x": 1}, config={"configurable": {"thread_id": "t1"}})
# {'x': 1, '__interrupt__': [{'question': 'هل أوافق؟'}]}
out2 = app.invoke(Command(resume="نعم"), config={"configurable": {"thread_id": "t1"}})
```

### وكيل ReAct جاهز
```python
from pygraph import create_react_agent
app = create_react_agent(model, tools)
app.invoke({"messages": [{"role": "user", "content": "hello"}]})
```

### Functional API
```python
from pygraph import entrypoint, task
@task
def double(x: int): return x * 2

@entrypoint()
def flow(inputs): return {"y": double(inputs["x"]) + 1}
flow.invoke({"x": 3})  # {'y': 7}
```

### حفظ دائم
```python
from pygraph import SqliteSaver
with SqliteSaver("ckpts.db") as sv:
    app = g.compile(checkpointer=sv)
```

## لماذا أسرع؟

- **بدون pydantic**: الحالة `dict` مباشرة، والـ reducers تُستخرج مرة واحدة عند البناء.
- **fast-path خطي**: العقدة الواحدة تُنفذ مباشرة بدون ThreadPool وبدون نسخ حالة للـ routing إلا عند وجود `conditional`.
- **static edges** محسوبة عند `compile`.
- **deepcopy فقط عند checkpoint** (`MemorySaver`/`SqliteSaver`)، والـ stream يعيد نفس الكائنات بدون نسخ.
- **fan-out متوازٍ**: `Send` والطوابير المتفرعة عبر `ThreadPoolExecutor` (و `asyncio.gather` في المسار غير المتزامن)، و `batch` متوازٍ، و `ToolNode` ينفذ الأدوات متوازياً.

نتيجة `benchmarks/bench_pygraph.py` (Windows، 50 عقدة × 300):

- `pygraph`: ~259k عقدة/ثا — أسرع ~1.5x من baseline ثقيل (deepcopy + سجل + تحقق كل خطوة بأسلوب langgraph).
- `fan-out ×8` مع I/O (2ms لكل عامل): ~6ms متوازية بدل ~16ms متسلسلة.

## اختبارات

```bash
python -m pytest tests/test_pygraph.py -q        # 17 اختباراً
python benchmarks/bench_pygraph.py
```
