from __future__ import annotations

from coreot.workflows.step_registry import DEFAULT_STEP_REGISTRY
from coreot.workflows.step_registry import OPTIONAL_STEP_REGISTRY
from coreot.workflows.step_registry import StepDefinition


class WorkflowError(ValueError):
    """Raised when a workflow request is inconsistent with the registry."""


def ordered_step_names() -> list[str]:
    return list(DEFAULT_STEP_REGISTRY)


def optional_step_names() -> list[str]:
    return list(OPTIONAL_STEP_REGISTRY)


def runnable_step_names() -> list[str]:
    return ordered_step_names() + optional_step_names()


def get_step_definition(step_name: str) -> StepDefinition:
    try:
        return DEFAULT_STEP_REGISTRY[step_name]
    except KeyError as exc:
        try:
            return OPTIONAL_STEP_REGISTRY[step_name]
        except KeyError:
            known = ", ".join(runnable_step_names())
            raise WorkflowError(f"Unknown step {step_name!r}. Known steps: {known}") from exc


def step_plan(through: str | None = None) -> list[StepDefinition]:
    if through is None:
        return list(DEFAULT_STEP_REGISTRY.values())

    get_step_definition(through)
    plan: list[StepDefinition] = []
    for name, definition in DEFAULT_STEP_REGISTRY.items():
        plan.append(definition)
        if name == through:
            break
    return plan
