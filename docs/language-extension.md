# Multi-Language Extension

Goal: extend the `methods/ours` code graph from Python to JavaScript/TypeScript,
Rust, Go, Java, PHP, C/C++, and C# without changing the Graph Plane or Document
Plane contracts. Each language gets one vertical slice before repository coverage
is expanded.

## 1. Current State

- `LanguageAnalyzer`, `CodeGraphBuilder`, and the multi-language analyzers are
  implemented.
- Supported analyzers: Python, JavaScript, TypeScript/TSX, Rust, Go, Java, PHP,
  C, C++, and C#.
- File discovery, node/edge IDs, and the component/document pipeline are
  language-agnostic.
- `configs/repos.yaml` records the language of every benchmark repository.
- Python has full + update + rubric evaluations for GraphRAG and Rasa.
- Chart.js is checked out at the benchmark commit and its JS graph can be built;
  the full JS documentation experiment has not run.
- New languages have analyzer and graph tests but no document-generation or
  rubric evaluation yet.

## 2. `LanguageAnalyzer` Interface

Python-specific behavior is owned by analyzers; the builder only orchestrates:

```python
class LanguageAnalyzer(Protocol):
    name: str                    # "python", "javascript", ...
    extensions: tuple[str, ...]  # (".py",), (".js", ".jsx"), ...
    language: Language           # tree-sitter Language

    def parse_imports(self, source: bytes, root: Node, rel_path: str) -> list[ImportSpec]:
        ...

    def visit_definitions(
        self, source: bytes, root: Node, rel_path: str, file_id: str,
        info: FileInfo,
    ) -> None:
        ...

    def is_public(self, symbol: dict[str, Any], file: dict[str, Any]) -> bool:
        ...

    def resolve_call(self, callee: str, info: FileInfo, graph: GraphStore) -> str | None:
        ...
```

Constraints:

- `attrs.language` must be the actual language, never hard-coded `"python"`.
- Public API rules are language-specific (Python underscore rules, JS `export`,
  Java/C# `public`, ...).
- Import/call resolution prioritizes correct file-level edges; symbol-level calls
  may be conservative. Prefer fewer edges over false edges.
- Files that cannot be parsed are counted as `unparsed`; never drop them silently.
- Analyzers extract syntax, symbols, and relationships only; they never call an LLM.

## 3. Phases

### Phase A: Interface (Completed)

- `LanguageAnalyzer`, `CodeGraphBuilder`, and the analyzer registry.
- Python analyzer behavior preserved.
- All code nodes carry `language`.
- Tests cover language detection, symbols/APIs, imports, and call edges for every
  registered analyzer.

### Phase B: JS/TS Vertical Slice (Analyzer Complete; Experiment Not Started)

Candidate repositories:

- Chart.js (JavaScript, clear structure, commit `637228008562` prepared);
- marktext (JavaScript/Electron, larger; backup).

Required support:

- `.js/.jsx/.mjs/.cjs`, then `.ts/.tsx`;
- file-level edges for `import` / `export` / `require`;
- function/class/method/arrow symbols;
- `export` as the public API rule;
- skip `node_modules`, build output, and `dist`.

Acceptance:

- `repodoc generate` produces complete hierarchical docs without page compression;
- high-level pages still contain Mermaid;
- CodeWikiBench parse, rubric, and judge can run;
- token/cost/time recorded in `metrics.json`;
- compare with CodeWiki at the same commit, rubric, and judge.

### Phase C: Java/C# (Analyzer Complete; Experiment Not Started)

- Java: `public/protected`, class/interface/enum/record, package/import,
  constructors.
- C#: namespace/using, class/interface/record/struct, public API, property/field.
- Choose one medium-sized repository each; do not run the full core split yet.

### Phase D: C/C++ (Analyzer Complete; Experiment Not Started)

- Header/implementation pairing.
- `#include` graph, function declarations/definitions, struct/class, namespace,
  access specifiers.
- Preprocessor and macros are conservative to avoid false call edges.

### Phase E: Rust/Go/PHP (Analyzer Complete; Experiment Not Started)

- Rust: `use`/`mod`, struct/enum/trait/impl, `pub` API.
- Go: package imports, type/function/method, uppercase API rule.
- PHP: namespace/use/require, class/interface/trait/enum, public methods.

## 4. Experiment Records

For each new language, use:

```text
runs/<language>-<repo>-full-v1/systems/{ours,codewiki}/<repo>/
```

Run one medium-sized repository per language for the first comparison. Expand only
after that slice is stable. Reports must include rubric score, page count, tokens,
cost, wall time, and LLM time.

## 5. Risks

- tree-sitter grammar and API version differences;
- JS/TS module resolution (path aliases, monorepos, dynamic imports) can create
  false edges;
- C/C++ macros and templates substantially increase parsing complexity;
- page counts are not comparable across languages; rubric and architecture-page
  quality are the primary signals.
