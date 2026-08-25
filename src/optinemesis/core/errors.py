"""Exception taxonomy for OptiNemesis."""


class OptinemesisError(Exception):
    """Base class for all OptiNemesis errors."""


class ContractError(OptinemesisError):
    """Raised when a contract, specification, or configuration is invalid."""


class BudgetExhausted(OptinemesisError):
    """Raised by CountingObjective when the evaluation budget is spent.

    Adapters must catch this exception, retain partial state, and return a
    RunResult. Silent overshoot is forbidden.
    """

    def __init__(self, budget: int, consumed: int) -> None:
        self.budget = budget
        self.consumed = consumed
        super().__init__(
            f"Evaluation budget exhausted: {consumed} of {budget} allowed "
            f"evaluations consumed."
        )


class RegistryError(OptinemesisError):
    """Raised on unknown or duplicate registrations."""


class AdapterError(OptinemesisError):
    """Raised when an optimizer adapter violates its contract."""
