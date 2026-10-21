"""Entry-point detection (routes / CLI commands / HTTP handlers)."""

from repodoc.agent.update import ChangeType, DiffSpec, analyze_diff
from repodoc.graph.analyzers import analyze_file


def test_python_flask_and_click_entry_points():
    code = (
        "from flask import Flask\n"
        "import click\n"
        "app = Flask(__name__)\n"
        "@app.get('/users')\n"
        "def list_users():\n"
        "    return []\n"
        "@click.command()\n"
        "def sync():\n"
        "    pass\n"
    )
    nodes, _rels, _imports = analyze_file("app.py", code, ".")
    by_name = {n.name: n for n in nodes}
    assert by_name["list_users"].entry_point == "HTTP GET /users"
    assert by_name["sync"].entry_point == "CLI command"


def test_java_spring_mapping_entry_point():
    code = (
        "package demo;\n"
        "public class UserController {\n"
        "  @GetMapping(\"/users/{id}\")\n"
        "  public User get(long id) { return null; }\n"
        "}\n"
    )
    nodes, _rels, _imports = analyze_file("demo/UserController.java", code, ".")
    method = next(n for n in nodes if n.component_type == "method")
    assert method.entry_point == "HTTP GET /users/{id}"


def test_go_handlefunc_entry_point():
    code = (
        "package main\n"
        "import \"net/http\"\n"
        "func handler(w http.ResponseWriter, r *http.Request) {}\n"
        "func main() { http.HandleFunc(\"/users\", handler) }\n"
    )
    nodes, _rels, _imports = analyze_file("main.go", code, ".")
    handler = next(n for n in nodes if n.name == "handler")
    assert handler.entry_point == "HTTP handler /users"


def test_entry_point_change_is_a_severe_api_change():
    old = "from flask import Flask\napp = Flask(__name__)\n@app.get('/a')\ndef view():\n    return 1\n"
    new = old.replace("/a", "/b")
    analysis = analyze_diff("app.py", old, new, DiffSpec("python"))
    change = next(c for c in analysis.changes if c.name == "view")
    assert change.change_type == ChangeType.API_SIGNATURE_CHANGED
    assert "/b" in (change.new_signature or "")
