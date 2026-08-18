"""Compatibility and CLI wrapper for ``nsrom.plotting.speedup``."""

if __name__ == "__main__":
    from runpy import run_module

    run_module("nsrom.plotting.speedup", run_name="__main__")
else:
    from nsrom.plotting.speedup import *  # noqa: F401,F403
