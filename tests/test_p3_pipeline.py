"""P3: system-role LLM calls, structured summaries, clustering cache/schema."""

from repodoc.config import Config
from repodoc.llm import LLMClient
from repodoc.utils import summarize_markdown


# --- LLM role separation ----------------------------------------------------


def test_build_messages_puts_instructions_in_system_role():
    messages = LLMClient.build_messages("user body", "system rules")
    assert messages[0] == {"role": "system", "content": "system rules"}
    assert messages[1] == {"role": "user", "content": "user body"}

    only_user = LLMClient.build_messages("user body", None)
    assert only_user == [{"role": "user", "content": "user body"}]

    blank_system = LLMClient.build_messages("user body", "   ")
    assert blank_system == [{"role": "user", "content": "user body"}]


def test_system_prompts_carry_injection_guard():
    from repodoc.agent.prompts import format_component_prompt

    system, _user = format_component_prompt(
        component_name="A",
        component_type="class",
        file_path="a.py",
        docstring="",
        source_code="class A: pass",
        dependencies=[],
        predecessors=[],
        module_name="pkg",
    )
    assert "untrusted DATA" in system


# --- Structured summaries ---------------------------------------------------


def test_generated_artifacts_are_world_readable(tmp_path):
    import stat

    from repodoc.graph.graph import HeterogeneousGraph
    from repodoc.graph.models import CodeNode
    from repodoc.utils import file_manager

    payload = tmp_path / "meta.json"
    file_manager.save_json({"a": 1}, str(payload))
    assert stat.S_IMODE(payload.stat().st_mode) == 0o644

    graph = HeterogeneousGraph()
    graph.add_code_node(
        CodeNode(id="a", type="function", component_type="function", file_path="a.py")
    )
    graph_path = tmp_path / "graph.json"
    graph.save(graph_path)
    assert stat.S_IMODE(graph_path.stat().st_mode) == 0o644


def test_summarize_short_doc_is_unchanged():
    doc = "# Title\n\nshort body"
    assert summarize_markdown(doc, budget=800) == doc


def test_summarize_long_doc_keeps_structure_within_budget():
    doc = (
        "# Title\n\nOpening paragraph line one. " + "word " * 500 + "\n\n"
        "## Section A\n\n- point one\n- point two\n\n"
        "## Section B\n\n" + "x" * 3000
    )
    summary = summarize_markdown(doc, budget=800)
    assert summary.startswith("# Title")
    assert "Opening paragraph line one." in summary
    assert "## Section A" in summary
    assert "point one" in summary
    assert len(summary) <= 800


# --- Clustering cache / schema / stats --------------------------------------


def _config(tmp_path):
    return Config(repo_path=str(tmp_path), output_dir=str(tmp_path / "out"))


def test_cluster_cache_avoids_duplicate_llm_calls(tmp_path, monkeypatch):
    import repodoc.graph.cluster as cluster

    calls = []
    response = '{"core": {"path": "src", "components": ["src.mod.f"]}}'

    def fake_call_llm(prompt, config, **kwargs):
        calls.append(prompt)
        return response, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(cluster, "call_llm", fake_call_llm)
    monkeypatch.setattr(cluster, "_cluster_cache_mem", None)
    config = _config(tmp_path)

    first = cluster._call_llm_with_retry("prompt A", config)
    second = cluster._call_llm_with_retry("prompt A", config)
    assert len(calls) == 1
    assert first[1] == second[1]

    cluster._call_llm_with_retry("prompt B", config)
    assert len(calls) == 2

    # Cache key includes the model: a different model re-asks.
    other = _config(tmp_path)
    other.cluster_model = "another-model"
    cluster._call_llm_with_retry("prompt A", other)
    assert len(calls) == 3


def test_parse_llm_response_sanitizes_malformed_shapes():
    import repodoc.graph.cluster as cluster

    tree = cluster._parse_llm_response(
        '{"good": {"path": 1, "components": "src.mod.f"}, "bad": "oops"}'
    )
    assert "bad" not in tree
    assert tree["good"]["components"] == ["src.mod.f"]
    assert tree["good"]["path"] == ""


def test_tree_sitter_parsers_are_reused():
    from repodoc.graph.analyzers import go, javascript, typescript

    assert javascript._get_js_parser() is javascript._get_js_parser()
    assert go._get_go_parser() is go._get_go_parser()
    ts_parser, _ = typescript._get_ts_parser(False)
    tsx_parser, _ = typescript._get_ts_parser(True)
    assert ts_parser is typescript._get_ts_parser(False)[0]
    assert tsx_parser is not ts_parser


def test_module_tree_stats(tmp_path):
    import repodoc.graph.cluster as cluster

    tree = {
        "core": {
            "path": "src",
            "components": ["a", "b"],
            "children": {
                "io": {"path": "src/io", "components": ["c"], "children": {}}
            },
        },
        "empty": {"path": "x", "components": [], "children": {}},
    }
    stats = cluster.module_tree_stats(tree)
    assert stats["modules"] == 3
    assert stats["max_depth"] == 2
    assert stats["empty_modules"] == 1
    assert stats["largest_module_components"] == 2
    assert stats["leaf_components"] == 1  # only the io leaf's "c"
