"""Shared exceptions for staff-level OMR protocol validation."""


class ProtocolError(ValueError):
    """Raised when an input violates a versioned staff OMR contract."""
