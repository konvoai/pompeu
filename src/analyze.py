"""Backward-compatible entry point forwarding to `pompeu_fabra.analysis`."""

from pompeu_fabra.analysis import main

__all__ = ["main"]


if __name__ == "__main__":
    main()

