"""Compatibility imports for the preserved supervisor and immutable IO."""
class ContractError(RuntimeError):pass
def need(condition,reason):
    if not condition:raise ContractError(reason)
