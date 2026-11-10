"""Supported folder-based Flex optimizer; implementation lives in facet.optimization."""

from facet.optimization import search as _implementation

# Preserve imports used by the lifecycle/alignment diagnostic examples.
globals().update({name: getattr(_implementation, name) for name in dir(_implementation) if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
