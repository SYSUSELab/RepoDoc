"""Configuration loading and path resolution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .util import ROOT, HarnessError, read_json, resolve_path


class ConfigError(HarnessError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return data


def _expand(value: Any, root: Path) -> Any:
    """Replace {root} and expand ~ in config strings."""
    if isinstance(value, str):
        return os.path.expanduser(value.replace("{root}", str(root)))
    if isinstance(value, list):
        return [_expand(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, root) for key, item in value.items()}
    return value


class Config:
    def __init__(
        self,
        root: Path,
        config_dir: Path,
        systems: dict[str, Any],
        repos: dict[str, Any],
        evaluation: dict[str, Any],
        experiments: dict[str, Any],
        locks: dict[str, Any],
    ) -> None:
        self.root = root
        self.config_dir = config_dir
        self.systems_raw = systems
        self.repos_raw = repos
        self.evaluation = evaluation
        self.experiments_raw = experiments
        self.locks = locks

    @classmethod
    def load(cls, root: Path = ROOT, config_dir: Path | None = None) -> "Config":
        root = root.resolve()
        if config_dir is None:
            config_dir = Path(os.environ.get("REPODOC_CONFIG_DIR", root / "configs"))
        config_dir = Path(config_dir)
        if not config_dir.is_absolute():
            config_dir = root / config_dir
        config_dir = config_dir.resolve()

        systems = _expand(
            _load_yaml(config_dir / "systems.yaml").get("systems", {}), root
        )
        repos_doc = _expand(_load_yaml(config_dir / "repos.yaml"), root)
        evaluation = _expand(_load_yaml(config_dir / "evaluation.yaml"), root)
        experiments = _expand(
            _load_yaml(config_dir / "experiments.yaml").get("suites", {}), root
        )
        locks = read_json(root / "data" / "locks" / "repos.json", default={}) or {}

        cfg = cls(
            root=root,
            config_dir=config_dir,
            systems=systems,
            repos=repos_doc,
            evaluation=evaluation,
            experiments=experiments,
            locks=locks,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not isinstance(self.systems_raw, dict) or not self.systems_raw:
            raise ConfigError("configs/systems.yaml: no systems defined")
        if not isinstance(self.repos_raw, dict) or not isinstance(
            self.repos_raw.get("repos"), dict
        ):
            raise ConfigError(
                "configs/repos.yaml: expected a top-level `repos` mapping"
            )
        if not isinstance(self.evaluation, dict):
            raise ConfigError("configs/evaluation.yaml: expected a mapping")
        if self.experiments_raw is None:
            self.experiments_raw = {}
        if not isinstance(self.experiments_raw, dict):
            raise ConfigError("configs/experiments.yaml: `suites` must be a mapping")

    # ------------------------------------------------------------------
    # Systems
    # ------------------------------------------------------------------
    def system(self, name: str, *, allow_disabled: bool = False) -> dict[str, Any]:
        try:
            spec = dict(self.systems_raw[name])
        except KeyError as exc:
            known = ", ".join(sorted(self.systems_raw))
            raise ConfigError(f"unknown system '{name}' (known: {known})") from exc
        if not allow_disabled and spec.get("enabled", True) is False:
            raise ConfigError(
                f"system '{name}' is disabled in configs/systems.yaml; "
                "set enabled: true once it is implemented"
            )
        if not spec.get("source") or not spec.get("adapter"):
            raise ConfigError(f"system '{name}' needs both `source` and `adapter`")
        spec["name"] = name
        spec["source_path"] = resolve_path(spec["source"], self.root)
        spec["adapter_path"] = resolve_path(spec["adapter"], self.root)
        spec.setdefault("extra_args", [])
        spec.setdefault("env", {})
        return spec

    def system_names(self, *, include_disabled: bool = False) -> list[str]:
        names = []
        for name, spec in self.systems_raw.items():
            if include_disabled or spec.get("enabled", True) is not False:
                names.append(name)
        return sorted(names)

    def system_env(self, spec: dict[str, Any]) -> dict[str, str]:
        env = {str(k): str(v) for k, v in (spec.get("env") or {}).items()}
        return env

    # ------------------------------------------------------------------
    # Repositories
    # ------------------------------------------------------------------
    @property
    def repo_defaults(self) -> dict[str, Any]:
        return dict(self.repos_raw.get("defaults", {}))

    def repo(self, name: str) -> dict[str, Any]:
        repos = self.repos_raw.get("repos", {})
        try:
            spec = dict(repos[name])
        except KeyError as exc:
            known = ", ".join(repos)
            raise ConfigError(f"unknown repository '{name}' (known: {known})") from exc
        merged = {**self.repo_defaults, **spec}
        merged["name"] = name
        return merged

    def repo_names(self, split: str | None = None) -> list[str]:
        names = list(self.repos_raw.get("repos", {}).keys())
        if split is None:
            return names
        return [name for name in names if self.repo(name).get("split") == split]

    def repo_commit(self, name: str, ref: str = "base") -> str | None:
        if ref not in ("base", "update"):
            raise ConfigError(f"ref must be 'base' or 'update' (got: {ref})")
        key = "commit" if ref == "base" else "update_commit"
        spec = self.repo(name)
        if spec.get(key):
            return str(spec[key])
        lock = self.locks.get(name, {}) if isinstance(self.locks, dict) else {}
        if lock.get(key):
            return str(lock[key])
        if ref == "base" and lock.get("commit"):
            return str(lock["commit"])
        return None

    def repo_url(self, name: str) -> str | None:
        spec = self.repo(name)
        if spec.get("url"):
            return str(spec["url"])
        lock = self.locks.get(name, {}) if isinstance(self.locks, dict) else {}
        return lock.get("url")

    @property
    def repos_dir(self) -> Path:
        return self.root / "data" / "repos"

    @property
    def checkouts_dir(self) -> Path:
        return self.root / "data" / "checkouts"

    def clone_dir(self, name: str) -> Path:
        return self.repos_dir / name

    def checkout_dir(self, name: str, ref: str = "base") -> Path:
        commit = self.repo_commit(name, ref)
        if not commit:
            return self.repos_dir / name
        return self.checkouts_dir / name / f"{ref}-{commit[:12]}"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def reference_root(self) -> Path:
        return self.data_dir / "reference"

    def reference_parsed_dir(self, name: str) -> Path:
        return self.reference_root / name / "parsed"

    @property
    def rubric_root(self) -> Path:
        return resolve_path(
            self.evaluation.get("rubrics", {}).get("dir", "data/rubrics"), self.root
        )

    def rubric_path(self, name: str) -> Path:
        return self.rubric_root / name / "combined_rubrics.json"

    @property
    def runs_dir(self) -> Path:
        return resolve_path(
            self.evaluation.get("run", {}).get("experiments_dir", "runs"), self.root
        )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    @property
    def benchmark_source(self) -> Path:
        return resolve_path(
            self.evaluation.get("benchmark", {}).get(
                "source", "benchmarks/codewikibench"
            ),
            self.root,
        )

    @property
    def bench_python(self) -> str:
        env_python = os.environ.get("REPODOC_BENCH_PYTHON")
        value = env_python or self.evaluation.get("benchmark", {}).get("python")
        if not value:
            return sys.executable
        value = str(value)
        if os.sep in value or value.startswith("~"):
            path = Path(os.path.expanduser(value))
            if not path.is_absolute():
                path = self.root / path
            # Do not resolve symlinks: a venv's bin/python is a symlink, and
            # resolving it makes Python use the base interpreter instead.
            return str(path.absolute())
        return value

    def bench_env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(base if base is not None else os.environ)
        paths: list[str] = []
        source_src = self.benchmark_source / "src"
        if source_src.is_dir():
            paths.append(str(source_src))
        for item in self.evaluation.get("benchmark", {}).get("pythonpath", []) or []:
            paths.append(str(resolve_path(item, self.root)))
        old = env.get("PYTHONPATH")
        if old:
            paths.append(old)
        if paths:
            env["PYTHONPATH"] = os.pathsep.join(paths)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    def bench_module_available(self) -> tuple[bool, str]:
        if not self.benchmark_source.is_dir():
            return False, f"benchmark checkout missing: {self.benchmark_source}"
        proc_env = self.bench_env()
        try:
            proc = subprocess.run(
                [
                    self.bench_python,
                    "-c",
                    "import codewikibench; print(getattr(codewikibench, '__file__', '?'))",
                ],
                cwd=str(self.benchmark_source),
                env=proc_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as exc:
            return False, f"cannot run benchmark Python {self.bench_python}: {exc}"
        return proc.returncode == 0, proc.stdout.strip()

    # ------------------------------------------------------------------
    # Suites
    # ------------------------------------------------------------------
    def suite(self, name: str) -> dict[str, Any]:
        try:
            spec = dict(self.experiments_raw[name])
        except KeyError as exc:
            known = ", ".join(sorted(self.experiments_raw))
            raise ConfigError(f"unknown suite '{name}' (known: {known})") from exc
        repos = spec.get("repos", [])
        if repos == "all":
            repos = self.repo_names()
        elif isinstance(repos, str):
            repos = [repos]
        systems = spec.get("systems", [])
        if isinstance(systems, str):
            systems = [systems]
        spec["repos"] = list(repos)
        spec["systems"] = list(systems)
        spec["name"] = name
        return spec

    def suite_names(self) -> list[str]:
        return sorted(self.experiments_raw)
