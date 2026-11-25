from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from repodoc.graph.builder import CodeGraphBuilder  # noqa: E402
from repodoc.graph.languages import analyzer_by_name, detect_analyzer  # noqa: E402


class LanguageAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="repodoc-lang-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_detect_javascript_and_build_edges(self) -> None:
        repo = self.tmp / "js"
        repo.mkdir()
        (repo / "util.js").write_text(
            "export function add(a, b) { return a + b; }\n"
            "export class Calc { multiply(a, b) { return a * b; } }\n",
            encoding="utf-8",
        )
        (repo / "app.js").write_text(
            'import { add } from "./util.js";\n'
            "export function run(x) { return add(x, 2); }\n",
            encoding="utf-8",
        )
        analyzer = detect_analyzer(repo)
        self.assertEqual(analyzer.name, "javascript")
        graph = CodeGraphBuilder(repo, repo_name="js", analyzer=analyzer).build()
        files = {node["attrs"]["path"] for node in graph.nodes_of_type("file")}
        self.assertEqual(files, {"app.js", "util.js"})
        self.assertTrue(
            all(
                node["attrs"].get("language") == "javascript"
                for node in graph.nodes_of_type("file")
            )
        )
        api_names = {
            node["attrs"]["qualified_name"] for node in graph.nodes_of_type("api")
        }
        self.assertIn("add", api_names)
        self.assertIn("Calc", api_names)
        self.assertIn("run", api_names)
        self.assertEqual(len(graph.edges_of_type("imports")), 1)
        self.assertTrue(graph.edges_of_type("calls"))

    def test_detect_typescript(self) -> None:
        repo = self.tmp / "ts"
        repo.mkdir()
        (repo / "util.ts").write_text(
            "export interface Point { x: number; y: number }\n"
            "export function add(a: number, b: number): number { return a + b; }\n",
            encoding="utf-8",
        )
        analyzer = detect_analyzer(repo)
        self.assertEqual(analyzer.name, "typescript")
        graph = CodeGraphBuilder(repo, repo_name="ts", analyzer=analyzer).build()
        self.assertTrue(graph.nodes_of_type("file"))
        self.assertTrue(graph.nodes_of_type("api"))

    def test_detect_tsx(self) -> None:
        repo = self.tmp / "tsx"
        repo.mkdir()
        (repo / "app.tsx").write_text(
            "export const App = () => <div>hello</div>;\n", encoding="utf-8"
        )
        self.assertEqual(detect_analyzer(repo).name, "typescript")
        graph = CodeGraphBuilder(
            repo, repo_name="tsx", analyzer=detect_analyzer(repo)
        ).build()
        self.assertTrue(graph.nodes_of_type("file"))

    def _repo(self, name: str, files: dict[str, str]) -> Path:
        repo = self.tmp / name
        repo.mkdir()
        for rel, content in files.items():
            path = repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return repo

    def _build(self, name: str, files: dict[str, str]) -> tuple[object, object]:
        repo = self._repo(name, files)
        analyzer = detect_analyzer(repo)
        graph = CodeGraphBuilder(repo, repo_name=name, analyzer=analyzer).build()
        return analyzer, graph

    @staticmethod
    def _attrs(graph: object, kind: str) -> set[str]:
        return {node["attrs"]["qualified_name"] for node in graph.nodes_of_type(kind)}

    def test_rust_analyzer(self) -> None:
        analyzer, graph = self._build(
            "rust",
            {
                "src/util.rs": "pub fn helper() {}\nfn hidden() {}\n",
                "src/lib.rs": (
                    "mod util;\n"
                    "use crate::util::helper;\n"
                    "pub struct Foo { x: i32 }\n"
                    "impl Foo {\n"
                    "  pub fn run(&self) { helper(); self.hidden(); }\n"
                    "  fn hidden(&self) {}\n"
                    "}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "rust")
        self.assertEqual(
            {node["attrs"]["language"] for node in graph.nodes_of_type("file")},
            {"rust"},
        )
        self.assertTrue(
            {"Foo", "Foo.run", "Foo.hidden", "helper"} <= self._attrs(graph, "symbol")
        )
        self.assertIn("Foo.run", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::Foo.run") and edge["dst"].endswith("::helper")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_go_analyzer(self) -> None:
        analyzer, graph = self._build(
            "go",
            {
                "go.mod": "module example.com/demo\n",
                "pkg/util/util.go": (
                    "package util\nfunc Helper() {}\nfunc hidden() {}\n"
                ),
                "main.go": (
                    "package main\n"
                    'import "example.com/demo/pkg/util"\n'
                    "type Foo struct { X int }\n"
                    "func NewFoo() *Foo { return &Foo{} }\n"
                    "func (f *Foo) Run() { f.helper(); util.Helper() }\n"
                    "func (f *Foo) helper() {}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "go")
        self.assertTrue(
            {"Foo", "NewFoo", "Foo.Run", "Helper"} <= self._attrs(graph, "symbol")
        )
        self.assertTrue(
            {"Foo", "NewFoo", "Foo.Run", "Helper"} <= self._attrs(graph, "api")
        )
        self.assertTrue(graph.edges_of_type("imports"))
        calls = {
            (edge["src"].split("::")[-1], edge["dst"].split("::")[-1])
            for edge in graph.edges_of_type("calls")
        }
        self.assertIn(("Foo.Run", "Foo.helper"), calls)
        self.assertIn(("Foo.Run", "Helper"), calls)

    def test_java_analyzer(self) -> None:
        base = "src/main/java/com/example/"
        analyzer, graph = self._build(
            "java",
            {
                base + "Util.java": (
                    "package com.example;\n"
                    "public class Util {\n"
                    "  public static void help() {}\n"
                    "}\n"
                ),
                base + "App.java": (
                    "package com.example;\n"
                    "import com.example.Util;\n"
                    "public class App {\n"
                    "  public App() {}\n"
                    "  public void run() { Util.help(); this.hidden(); }\n"
                    "  private void hidden() {}\n"
                    "}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "java")
        self.assertTrue(
            {"App", "App.App", "App.run", "Util", "Util.help"}
            <= self._attrs(graph, "symbol")
        )
        self.assertIn("App.run", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::App.run")
                and edge["dst"].endswith("::Util.help")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_php_analyzer(self) -> None:
        analyzer, graph = self._build(
            "php",
            {
                "src/Util.php": (
                    "<?php\n"
                    "namespace App;\n"
                    "class Util { public static function help(): void {} }\n"
                ),
                "src/App.php": (
                    "<?php\n"
                    "namespace App;\n"
                    "use App\\Util;\n"
                    "class App {\n"
                    "  public function run(): void { Util::help(); $this->hidden(); }\n"
                    "  private function hidden(): void {}\n"
                    "}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "php")
        self.assertTrue(
            {"App", "App.run", "Util", "Util.help"} <= self._attrs(graph, "symbol")
        )
        self.assertIn("App.run", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::App.run")
                and edge["dst"].endswith("::Util.help")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_c_analyzer(self) -> None:
        analyzer, graph = self._build(
            "c",
            {
                "include/util.h": "int add(int a, int b);\n",
                "src/main.c": (
                    '#include "util.h"\nint main(void) { add(1, 2); return 0; }\n'
                ),
            },
        )
        self.assertEqual(analyzer.name, "c")
        self.assertIn("add", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::main") and edge["dst"].endswith("::add")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_cpp_analyzer(self) -> None:
        analyzer, graph = self._build(
            "cpp",
            {
                "src/foo.hpp": (
                    "#pragma once\n"
                    "class Foo { public: void Method(); private: void Hidden(); };\n"
                ),
                "src/foo.cpp": (
                    '#include "foo.hpp"\n'
                    "void Foo::Method() { Hidden(); }\n"
                    "void Foo::Hidden() {}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "cpp")
        self.assertTrue(
            {"Foo", "Foo.Method", "Foo.Hidden"} <= self._attrs(graph, "symbol")
        )
        self.assertIn("Foo.Method", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::Foo.Method")
                and edge["dst"].endswith("::Foo.Hidden")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_csharp_analyzer(self) -> None:
        analyzer, graph = self._build(
            "csharp",
            {
                "src/Util.cs": (
                    "namespace App.Services;\n"
                    "public static class Util { public static void Help() {} }\n"
                ),
                "src/App.cs": (
                    "using Alias = App.Services.Util;\n"
                    "namespace App;\n"
                    "public class App {\n"
                    "  public void Run() { Alias.Help(); this.Hidden(); }\n"
                    "  private void Hidden() {}\n"
                    "}\n"
                ),
            },
        )
        self.assertEqual(analyzer.name, "csharp")
        self.assertTrue(
            {"App", "App.Run", "Util", "Util.Help"} <= self._attrs(graph, "symbol")
        )
        self.assertIn("App.Run", self._attrs(graph, "api"))
        self.assertTrue(graph.edges_of_type("imports"))
        self.assertTrue(
            any(
                edge["src"].endswith("::App.Run")
                and edge["dst"].endswith("::Util.Help")
                for edge in graph.edges_of_type("calls")
            )
        )

    def test_detection_prefers_cpp_over_header_only_c(self) -> None:
        repo = self._repo(
            "cpp-header",
            {
                "src/main.cpp": "int main() { return 0; }\n",
                "src/main.h": "int main();\n",
            },
        )
        self.assertEqual(detect_analyzer(repo).name, "cpp")
        c_repo = self._repo(
            "c-header",
            {
                "src/main.c": "int main(void) { return 0; }\n",
                "src/main.h": "int main(void);\n",
            },
        )
        self.assertEqual(detect_analyzer(c_repo).name, "c")

    def test_analyzer_aliases(self) -> None:
        for alias, expected in {
            "rs": "rust",
            "golang": "go",
            "java": "java",
            "php": "php",
            "c++": "cpp",
            "csharp": "csharp",
            "cs": "csharp",
        }.items():
            self.assertEqual(analyzer_by_name(alias).name, expected)


if __name__ == "__main__":
    unittest.main()
