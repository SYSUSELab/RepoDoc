"""RepoDoc configuration."""

from dataclasses import dataclass
import os

# Token limit for clustering: skip LLM cluster when content is under this (CodeWiki default)
MAX_TOKEN_PER_MODULE = 36_369

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
    # Optional LLM-based concept extraction (enhances keyword matching)
    llm_concept_extraction: bool = False  # set True for LLM-enhanced concept extraction
    llm_concept_max_components: int = 100  # max components to process with LLM (cost control)
    # Ablation switches: cluster_strategy / doc_strategy
    cluster_strategy: str = "llm"  # "llm" | "heuristic"
    doc_strategy: str = "agent"  # "agent" | "direct"
    # Resumable generation: when False (default), docs that are already
    # generated (graph describes edge + file on disk) are skipped on re-runs.
    force_regenerate: bool = False

    def __post_init__(self) -> None:
        self.llm_base_url = _strip_chat_completions_suffix(self.llm_base_url)
        if self.cluster_model is None:
            self.cluster_model = self.main_model
        if self.dependency_graph_dir is None:
            self.dependency_graph_dir = os.path.join(self.output_dir, "data")
        if self.docs_dir is None:
            from repodoc.utils import project_name

            self.docs_dir = os.path.join(
                self.output_dir, "docs", f"{project_name(self.repo_path)}-docs"
            )
        if self.cluster_strategy not in ("llm", "heuristic"):
            raise ValueError(
                f"cluster_strategy must be 'llm' or 'heuristic', got {self.cluster_strategy!r}"
            )
        if self.doc_strategy not in ("agent", "direct"):
            raise ValueError(
                f"doc_strategy must be 'agent' or 'direct', got {self.doc_strategy!r}"
            )
