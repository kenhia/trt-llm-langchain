"""Tiny control-plane CLI: ``trtllm-lc <list|status|load|unload|ensure> [model-key]``.

A Sprint-1 smoke tool for driving the backend without LangChain. Talks only to the KServe
control plane (``control_url``); does not require the OpenAI proxy.
"""

from __future__ import annotations

import argparse
import sys

from .config import TrtLlmSettings
from .errors import TrtLlmError
from .manager import TrtLlmManager


def _cmd_list(mgr: TrtLlmManager) -> int:
    for key in mgr.available_keys():
        print(key)
    return 0


def _cmd_status(mgr: TrtLlmManager) -> int:
    if not mgr.is_healthy():
        print("backend: DOWN (control plane unreachable)", file=sys.stderr)
        return 1
    models = mgr.models()
    print(f"{'MODEL':<32} {'KIND':<9} {'ENSEMBLE':<12} RESPONSIVE")
    for key in sorted(models):
        info = models[key]
        kind = "vision" if info.is_vision else "chat"
        ensemble = info.components.get("ensemble", "-")
        responsive = "yes" if (info.loaded and mgr.is_responsive(key)) else "no"
        print(f"{key:<32} {kind:<9} {ensemble:<12} {responsive}")
    return 0


def _cmd_load(mgr: TrtLlmManager, key: str) -> int:
    print(f"Loading {key} ...")
    mgr.load(key)
    print(f"  {key} loaded; responsive={mgr.is_responsive(key)}")
    return 0


def _cmd_unload(mgr: TrtLlmManager, key: str) -> int:
    print(f"Unloading {key} ...")
    mgr.unload(key)
    print(f"  {key} unloaded")
    return 0


def _cmd_ensure(mgr: TrtLlmManager, key: str) -> int:
    print(f"Ensuring {key} is resident ...")
    mgr.ensure_loaded(key)
    print(f"  {key} responsive={mgr.is_responsive(key)}; loaded={mgr.loaded_keys()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trtllm-lc", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list model keys the backend knows about")
    sub.add_parser("status", help="show load/responsive state for all models")
    for name in ("load", "unload", "ensure"):
        p = sub.add_parser(name, help=f"{name} a model by key")
        p.add_argument("model", help="model key, e.g. qwen2_5-coder-7b-fp16")
    args = parser.parse_args(argv)

    mgr = TrtLlmManager(TrtLlmSettings.from_env())
    try:
        with mgr:
            if args.command == "list":
                return _cmd_list(mgr)
            if args.command == "status":
                return _cmd_status(mgr)
            if args.command == "load":
                return _cmd_load(mgr, args.model)
            if args.command == "unload":
                return _cmd_unload(mgr, args.model)
            if args.command == "ensure":
                return _cmd_ensure(mgr, args.model)
    except TrtLlmError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
