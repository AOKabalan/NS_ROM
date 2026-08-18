#!/usr/bin/env python3
"""Compatibility and CLI wrapper for ``nsrom.io.mass``."""

if __name__ == "__main__":
    from runpy import run_module

    run_module("nsrom.io.mass", run_name="__main__")
else:
    from nsrom.io.mass import *  # noqa: F401,F403
