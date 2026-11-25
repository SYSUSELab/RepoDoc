"""Environment and data readiness checks."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import Config
from .util import git_head, run_capture


class Doctor:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def add(self, level: str, label: str, detail: str = "") -> None:
        self.results.append((level, label, detail))

    def ok(self, label: str, detail: str = "") -> None:
        self.add("ok", label, detail)

    def warn(self, label: str, detail: str = "") -> None:
        self.add("warn", label, detail)

    def fail(self, label: str, detail: str = "") -> None:
        self.add("fail", label, detail)

    def report(self, *, strict: bool = False) -> int:
        for level, label, detail in self.results:
            prefix = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[fail]"}[level]
            print(f"{prefix} {label}" + (f"\n       {detail}" if detail else ""))
        failures = sum(1 for level, _, _ in self.results if level == "fail")
        warnings = sum(1 for level, _, _ in self.results if level == "warn")
        print(f"\n{len(self.results)} checks, {failures} failed, {warnings} warnings")
        return 1 if failures or (strict and warnings) else 0


def _python_version(executable: str) -> tuple[bool, str]:
    try:
        proc = run_capture(
            [
                executable,
                "-c",
                "import sys; print('.'.join(map(str, sys.version_info[:3])))",
            ]
        )
    except OSError as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, proc.stderr.strip() or "not runnable"
    return True, proc.stdout.strip()


def run_doctor(cfg: Config, *, strict: bool = False) -> int:
    doctor = Doctor()

    # Config and benchmark -------------------------------------------------
    doctor.ok("configuration", str(cfg.config_dir))
    if shutil.which("git"):
        doctor.ok("git", shutil.which("git") or "git")
    else:
        doctor.fail("git", "git is required to clone repositories and create worktrees")
    bench = cfg.benchmark_source
    if bench.is_dir():
        commit = git_head(bench)
        doctor.ok("benchmark checkout", f"{bench} @ {(commit or 'unknown')[:12]}")
    else:
        doctor.fail("benchmark checkout", f"missing: {bench}")

    bench_ok, detail = cfg.bench_module_available()
    if bench_ok:
        doctor.ok("codewikibench import", detail)
    else:
        doctor.fail(
            "codewikibench import",
            f"{cfg.bench_python} cannot import codewikibench. Run `make setup`.\n{detail}",
        )

    py_ok, py_version = _python_version(cfg.bench_python)
    if py_ok:
        try:
            major, minor = (int(x) for x in py_version.split(".")[:2])
        except ValueError:
            major, minor = 0, 0
        if (major, minor) >= (3, 10):
            doctor.ok("benchmark Python", f"{cfg.bench_python} ({py_version})")
        else:
            doctor.fail(
                "benchmark Python",
                f"{py_version} is too old; CodeWikiBench needs >=3.10",
            )
    else:
        doctor.fail("benchmark Python", f"{cfg.bench_python}: {py_version}")

    # Systems --------------------------------------------------------------
    for name in cfg.system_names(include_disabled=True):
        spec = cfg.system(name, allow_disabled=True)
        enabled = spec.get("enabled", True) is not False
        label = f"system {name}"
        if not enabled:
            doctor.warn(label, "disabled in configs/systems.yaml")
            continue
        if not spec["source_path"].is_dir():
            doctor.fail(label, f"source directory missing: {spec['source_path']}")
            continue
        if not spec["adapter_path"].is_file():
            doctor.fail(label, f"adapter missing: {spec['adapter_path']}")
            continue
        doctor.ok(label, f"{spec['adapter_path']}")

        if name == "codewiki":
            binary = os.environ.get("CODEWIKI_BIN") or cfg.system_env(spec).get(
                "CODEWIKI_BIN", "codewiki"
            )
            resolved = shutil.which(binary) or (
                binary if Path(binary).is_file() else None
            )
            if resolved:
                try:
                    proc = run_capture([resolved, "--version"])
                except OSError as exc:
                    doctor.fail("codewiki executable", f"cannot run {resolved}: {exc}")
                else:
                    version = (
                        proc.stdout.strip() or proc.stderr.strip()
                        if proc.returncode == 0
                        else None
                    )
                    doctor.ok("codewiki executable", version or resolved)
            else:
                doctor.fail(
                    "codewiki executable",
                    f"not found: {binary}. Run `make setup-codewiki` or edit configs/systems.yaml.",
                )

    # Data -----------------------------------------------------------------
    clones = list(cfg.repos_dir.glob("*")) if cfg.repos_dir.is_dir() else []
    checkouts = (
        list(cfg.checkouts_dir.glob("*/*")) if cfg.checkouts_dir.is_dir() else []
    )
    if clones:
        doctor.ok("repository clones", f"{len(clones)} under {cfg.repos_dir}")
    else:
        doctor.warn(
            "repository clones",
            f"none under {cfg.repos_dir}; run `python -m harness prepare --from-hf`",
        )

    if checkouts:
        doctor.ok("repository checkouts", f"{len(checkouts)} under {cfg.checkouts_dir}")
    else:
        doctor.warn("repository checkouts", "none; run `python -m harness prepare`")

    repo_names = cfg.repo_names()
    with_rubrics = [r for r in repo_names if cfg.rubric_path(r).is_file()]
    with_reference = [
        r
        for r in repo_names
        if (cfg.reference_parsed_dir(r) / "docs_tree.json").is_file()
        and (cfg.reference_parsed_dir(r) / "structured_docs.json").is_file()
    ]
    if with_rubrics:
        doctor.ok(
            "frozen rubrics", f"{len(with_rubrics)}/{len(repo_names)} repositories"
        )
    else:
        doctor.warn(
            "frozen rubrics", "none; download with `prepare --from-hf` or run `rubrics`"
        )
    if with_reference:
        doctor.ok(
            "reference docs", f"{len(with_reference)}/{len(repo_names)} repositories"
        )
    else:
        doctor.warn("reference docs", "none; run `prepare --from-hf`")

    # Environment ----------------------------------------------------------
    for key in ("BASE_URL", "API_KEY", "MODEL"):
        if os.environ.get(key):
            doctor.ok(f"env {key}", "set")
        else:
            doctor.warn(
                f"env {key}",
                "not set; CodeWikiBench has defaults but judges usually need an explicit endpoint",
            )
    if (cfg.root / ".env").is_file():
        doctor.ok(".env", str(cfg.root / ".env"))
    else:
        doctor.warn(
            ".env", "no project .env; copy .env.example if you use an API endpoint"
        )

    try:
        import datasets  # noqa: F401

        doctor.ok("datasets package", "installed (required only for prepare --from-hf)")
    except ImportError:
        doctor.warn(
            "datasets package", "not installed; needed only for prepare --from-hf"
        )

    return doctor.report(strict=strict)
