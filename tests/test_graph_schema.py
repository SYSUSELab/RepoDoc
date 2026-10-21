"""Knowledge-graph schema versioning."""

import json

import pytest

from repodoc.graph.graph import HeterogeneousGraph
from repodoc.graph.models import CodeNode, GRAPH_SCHEMA_VERSION


def test_graph_roundtrip_records_schema_version(tmp_path):
    graph = HeterogeneousGraph()
    graph.add_code_node(
        CodeNode(
            id="m.C",
            type="class",
            component_type="class",
            file_path="m.py",
            name="C",
            relative_path="m.py",
        )
    )
    path = tmp_path / "graph.json"
    graph.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == GRAPH_SCHEMA_VERSION
    loaded = HeterogeneousGraph.load(path)
    assert loaded.schema_version == GRAPH_SCHEMA_VERSION
    assert loaded.get_node("m.C").name == "C"


def test_legacy_graph_without_version_loads(tmp_path):
    payload = {
        "nodes": [
            {
                "id": "m.C",
                "node_type": "code",
                "data": {
                    "id": "m.C",
                    "type": "class",
                    "component_type": "class",
                    "file_path": "m.py",
                    "relative_path": "m.py",
                    "name": "C",
                },
            }
        ],
        "edges": [],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    loaded = HeterogeneousGraph.load(path)
    assert loaded.schema_version == 1
    assert loaded.get_node("m.C").name == "C"


def test_newer_schema_is_rejected(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(
        json.dumps(
            {"schema_version": GRAPH_SCHEMA_VERSION + 1, "nodes": [], "edges": []}
        )
    )
    with pytest.raises(ValueError):
        HeterogeneousGraph.load(path)
