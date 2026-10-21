"""RepoDoc configuration."""

from dataclasses import dataclass, field
import os

# Token limit for clustering: skip LLM cluster when content is under this (CodeWiki default)
MAX_TOKEN_PER_MODULE = 36_369

# Source files larger than this are skipped during analysis (generated /
# minified files only bloat the graph and the LLM prompts).
MAX_SOURCE_FILE_BYTES = 1_500_000

# The OpenAI SDK always appends "/chat/completions" to base_url, so a
# configured URL that already ends with it must be stripped (otherwise the
# request hits ".../chat/completions/chat/completions" -> 404).
_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"


def _strip_chat_completions_suffix(url: str) -> str:
    """Return the base URL without a trailing '/chat/completions' (no-op otherwise)."""
    stripped = url.rstrip("/")
    if stripped.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return stripped[: -len(_CHAT_COMPLETIONS_SUFFIX)]
    return url


@dataclass
class Config:
    """Configuration for RepoDoc."""

    repo_path: str
    output_dir: str = "output"
    dependency_graph_dir: str | None = None
    docs_dir: str | None = None
    max_depth: int = 2
    # LLM (Phase 2: clustering; Phase 3: doc generation)
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = "sk-1234"
    main_model: str = "deepseek-chat"
    cluster_model: str | None = None  # defaults to main_model
    # Extra HTTP headers for every LLM request (e.g. gateways that require a
    # session/routing header). Populated from LLM_EXTRA_HEADERS (JSON) by the
    # CLI's make_config.
    llm_headers: dict[str, str] = field(default_factory=dict)
    # Ablation switches: cluster_strategy / doc_strategy
    cluster_strategy: str = "llm"  # "llm" | "heuristic"
    doc_strategy: str = "agent"  # "agent" | "direct"
    # Analysis scope: only files matching include_patterns are analyzed;
    # exclude_patterns are added on top of the built-in ignore list.
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    max_file_size: int = MAX_SOURCE_FILE_BYTES
    # Documentation scope: empty doc_types keeps the automatic leaf selection;
    # otherwise only the listed component types are documented. max_docs caps
    # the count (0 = unlimited), ranked by popularity/type.
    doc_types: list[str] = field(default_factory=list)
    max_docs: int = 0
    # Language of generated documentation (prompt-level instruction).
    doc_language: str = "English"
    # Caller-propagation budget for update: cap the number of caller docs a
    # single change set can pull in (ranked by distance + fan-in). 0 = no cap.
    # Overridable via REPODOC_MAX_IMPACT_DOCS.
    max_impact_docs: int = 60

    def __post_init__(self) -> None:
        self.llm_base_url = _strip_chat_completions_suffix(self.llm_base_url)
        if self.cluster_model is None:
            self.cluster_model = self.main_model
        if self.dependency_graph_dir is None:
            self.dependency_graph_dir = os.path.join(self.output_dir, "data")
        if self.docs_dir is None:
            repo_name = os.path.basename(os.path.normpath(self.repo_path))
            safe_name = "".join(c if c.isalnum() else "_" for c in repo_name)
            self.docs_dir = os.path.join(self.output_dir, "docs", f"{safe_name}-docs")
        if self.cluster_strategy not in ("llm", "heuristic"):
            raise ValueError(
                f"cluster_strategy must be 'llm' or 'heuristic', got {self.cluster_strategy!r}"
            )
        if self.doc_strategy not in ("agent", "direct"):
            raise ValueError(
                f"doc_strategy must be 'agent' or 'direct', got {self.doc_strategy!r}"
            )
        if self.max_impact_docs < 0:
            raise ValueError(
                f"max_impact_docs must be >= 0, got {self.max_impact_docs!r}"
            )
