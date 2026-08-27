"""Failure-driven retry loop shared by doc generation and incremental update.

Both pipelines run many LLM-backed doc tasks concurrently and treat doc
completeness as the hard goal: when a batch has failures (network blips,
overloaded gateway), retry ONLY the failed jobs, and before each retry round
explicitly halve the shared adaptive limiter's concurrency so the retry
storm does not re-overload the gateway. This is the single implementation —
generate.py (component + module layers) and update.py all use it.
"""

import asyncio
from typing import Any, Awaitable, Callable


async def run_with_retry(
    jobs: list,
    run_fn: Callable[[Any], Awaitable],
    label: str,
    max_rounds: int = 4,
) -> tuple[list, list]:
    """Run jobs with a failure-driven concurrency-lowering retry loop.

    Each round retries only the failed jobs; before each retry the shared
    adaptive limiter's permits are explicitly halved (when above the minimum)
    so the retry does not re-overload the gateway. Stops early when a round
    has no failures or when concurrency bottomed out.

    Returns (results_aligned_to_jobs, still_failed_jobs). Results are the
    raw values from run_fn (SkillResult or Exception instances)."""
    from repodoc.llm import _LIMITER

    results = await asyncio.gather(
        *(run_fn(j) for j in jobs), return_exceptions=True
    )
    for _round in range(max_rounds):
        failed = [
            j
            for j, r in zip(jobs, results)
            if isinstance(r, Exception)
            or (r is not None and not getattr(r, "success", False))
        ]
        if not failed:
            break
        if _LIMITER.permits > _LIMITER.min_permits:
            _LIMITER.force_decrease()
        print(
            f"Concurrency {_LIMITER.permits}, retrying "
            f"{len(failed)} {label} (round {_round + 1})..."
        )
        retry = await asyncio.gather(
            *(run_fn(j) for j in failed), return_exceptions=True
        )
        # slot retry results back at the failed positions
        retry_iter = iter(retry)
        results = [
            next(retry_iter)
            if (
                isinstance(r, Exception)
                or (r is not None and not getattr(r, "success", False))
            )
            else r
            for r in results
        ]
        if _LIMITER.permits <= _LIMITER.min_permits:
            break  # concurrency bottomed out: this round was the last try

    still_failed = [
        j
        for j, r in zip(jobs, results)
        if isinstance(r, Exception)
        or (r is not None and not getattr(r, "success", False))
    ]
    return results, still_failed
