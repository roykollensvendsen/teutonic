"""Test harness fixtures for in-process simulation of validator deps.

Each module in this package replaces a single external dependency
(subtensor chain, eval_server SSE stream, validator main-loop tick)
with an in-process double whose contract is documented in the module's
docstring. Tests pin the docstring contract; what's NOT modeled is
called out explicitly so consumers don't reach for behavior the harness
never promised.
"""
