#!/usr/bin/env python3
"""
sync_paper.py -- copy generated artifacts into the paper repository.

The code repo generates; the paper repo consumes. Nothing under the paper's
generated/ is ever edited by hand, so a stale number cannot survive a
regeneration and nothing here needs to know LaTeX.

    python render/sync_paper.py                    # default paper path below
    python render/sync_paper.py /path/to/paper
    python render/sync_paper.py --dry-run

Run it after ANY renderer run. The macros are the numbers quoted in the prose,
so regenerating a figure without syncing leaves the text quoting the previous
run.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

DEFAULT_PAPER = os.path.expanduser("~/01_research/my_paper/Ali_Paper/paper")
SRC = "render/out"

NEWCOMMAND = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")
ANYDEF = re.compile(r"\\(?:newcommand|renewcommand|def)\s*\{?\\([A-Za-z]+)")
# an unescaped % starts a LaTeX comment; a definition inside one is not a
# definition, and treating it as one reports collisions that do not exist
COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def macro_names(paths, pattern):
    names = set()
    for p in paths:
        try:
            body = open(p, encoding="utf-8").read()
        except OSError:
            continue
        names.update(pattern.findall(COMMENT.sub("", body)))
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("paper", nargs="?", default=DEFAULT_PAPER)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"no {args.src} -- run the renderers first")
    if not os.path.isdir(args.paper):
        sys.exit(f"no such paper directory: {args.paper}")

    dests = {k: os.path.join(args.paper, k)
             for k in ("figures", "tables", "generated")}
    if not args.dry_run:
        for d in dests.values():
            os.makedirs(d, exist_ok=True)

    def entries(name):
        return sorted(f for f in os.listdir(args.src) if name(f))

    groups = [
        ("figures", "figures",
         lambda f: f.startswith("fig") and f.endswith(".pdf")),
        ("tables", "tables",
         lambda f: f.startswith("tab_") and f.endswith(".tex")),
        ("generated macros", "generated",
         lambda f: (f.startswith("macros_") or f == "section6_numbers.tex")
         and f.endswith(".tex")),
    ]

    print(f"paper: {args.paper}")
    copied = {}
    for label, key, pred in groups:
        print(f"\n--- {label} ---")
        names = entries(pred)
        copied[key] = names
        if not names:
            print("  (none)")
        for f in names:
            src = os.path.join(args.src, f)
            verb = "would copy" if args.dry_run else "copied"
            if not args.dry_run:
                shutil.copy2(src, os.path.join(dests[key], f))
            print(f"  {verb}  {f:<34s} -> {key}/")

    # A generated \newcommand that macros.tex also defines is a hard LaTeX error
    # ("command already defined"), and the message names the macro but not the
    # file. Catch it here rather than in an hour of log reading.
    print("\n--- collision check against macros.tex ---")
    macros_tex = os.path.join(args.paper, "macros.tex")
    if os.path.isfile(macros_tex):
        generated = macro_names(
            [os.path.join(args.src, f) for f in copied.get("generated", [])],
            NEWCOMMAND)
        hand = macro_names([macros_tex], ANYDEF)
        clashes = sorted(generated & hand)
        if clashes:
            print("  !! defined in BOTH macros.tex and the generated files.")
            print("     LaTeX will stop. Delete the hand-written definition --")
            print("     the generated one is the measured value:")
            for c in clashes:
                print(f"       \\{c}")
        else:
            print("  no collisions")
    else:
        print(f"  (macros.tex not found at {macros_tex} -- skipped)")

    # \CHECK{} is make_tables.py refusing to invent a number. It renders as
    # [CHECK: ...] in the build, which is the point -- but it should be seen
    # here first.
    print("\n--- unfilled cells ---")
    found = False
    for f in copied.get("tables", []):
        body = open(os.path.join(args.src, f), encoding="utf-8").read()
        n = body.count("CHECK")
        if n:
            print(f"       {f:<24s} {n} cell(s)")
            found = True
    if not found:
        print("  none")

    if copied.get("generated"):
        print("\n--- once, in the preamble AFTER \\input{macros} ---")
        print("  \\newcommand{\\CHECK}[1]{\\textbf{[CHECK: #1]}}")
        for f in copied["generated"]:
            print(f"  \\input{{generated/{f[:-4]}}}")


if __name__ == "__main__":
    main()
