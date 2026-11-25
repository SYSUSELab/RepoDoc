"""Dynamic token/context budgets derived from model limits."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import ModelLimits


@dataclass(frozen=True)
class BudgetPolicy:
    """Allocate a fraction of the model's limits, with safety floors.

    Ratios are primary; floors only prevent tiny models from receiving an
    unusable budget. Override ratios through environment variables or CLI flags.
    """

    context_ratio: float
    output_ratio: float
    min_input_tokens: int
    min_output_tokens: int
    chars_per_token: float = 3.5
    env_prefix: str = ""

    def input_tokens(self, limits: ModelLimits) -> int:
        raw = int(limits.context * self.context_ratio)
        value = max(self.min_input_tokens, raw)
        # Leave at least the output budget inside the model context window.
        value = min(value, max(1, limits.context - self.output_tokens(limits)))
        return value

    def output_tokens(self, limits: ModelLimits) -> int:
        raw = int(limits.output * self.output_ratio)
        return min(limits.output, max(self.min_output_tokens, raw))

    def context_chars(self, limits: ModelLimits) -> int:
        return int(self.input_tokens(limits) * self.chars_per_token)

    def with_env_overrides(self) -> "BudgetPolicy":
        if not self.env_prefix:
            return self
        context_ratio = float(
            os.environ.get(f"{self.env_prefix}_CONTEXT_RATIO", self.context_ratio)
        )
        output_ratio = float(
            os.environ.get(f"{self.env_prefix}_OUTPUT_RATIO", self.output_ratio)
        )
        return BudgetPolicy(
            context_ratio=context_ratio,
            output_ratio=output_ratio,
            min_input_tokens=self.min_input_tokens,
            min_output_tokens=self.min_output_tokens,
            chars_per_token=self.chars_per_token,
            env_prefix=self.env_prefix,
        )


@dataclass(frozen=True)
class ComputedBudget:
    limits: ModelLimits
    policy: BudgetPolicy
    input_tokens: int
    output_tokens: int
    context_chars: int

    def as_dict(self) -> dict[str, int | str | float]:
        return {
            "model": self.limits.model,
            "model_context": self.limits.context,
            "model_output": self.limits.output,
            "context_ratio": self.policy.context_ratio,
            "output_ratio": self.policy.output_ratio,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_chars": self.context_chars,
            "limits_source": self.limits.source,
        }


CLUSTER_POLICY = BudgetPolicy(
    context_ratio=0.02,
    output_ratio=0.02,
    min_input_tokens=2048,
    min_output_tokens=2048,
    env_prefix="REPODOC_CLUSTER",
)

DOC_POLICY = BudgetPolicy(
    context_ratio=0.04,
    output_ratio=0.05,
    min_input_tokens=4096,
    min_output_tokens=4096,
    env_prefix="REPODOC_DOC",
)


def policy_for(task: str) -> BudgetPolicy:
    if task == "cluster":
        return CLUSTER_POLICY.with_env_overrides()
    if task == "docs":
        return DOC_POLICY.with_env_overrides()
    raise ValueError(f"unknown budget task: {task}")


def compute_budget(
    limits: ModelLimits,
    task: str,
    *,
    context_ratio: float | None = None,
    output_ratio: float | None = None,
) -> ComputedBudget:
    policy = policy_for(task)
    if context_ratio is not None or output_ratio is not None:
        policy = BudgetPolicy(
            context_ratio=context_ratio
            if context_ratio is not None
            else policy.context_ratio,
            output_ratio=output_ratio
            if output_ratio is not None
            else policy.output_ratio,
            min_input_tokens=policy.min_input_tokens,
            min_output_tokens=policy.min_output_tokens,
            chars_per_token=policy.chars_per_token,
            env_prefix=policy.env_prefix,
        )
    return ComputedBudget(
        limits=limits,
        policy=policy,
        input_tokens=policy.input_tokens(limits),
        output_tokens=policy.output_tokens(limits),
        context_chars=policy.context_chars(limits),
    )
