"""umwelt-forge — the operator CLI for the embedded compiler + warden.

    umwelt-forge new greenhouse --rant "I have a small greenhouse with ..."
    umwelt-forge validate greenhouse
    umwelt-forge warden tick greenhouse
    umwelt-forge warden status greenhouse
    umwelt-forge warden promote greenhouse param_tune
    umwelt-forge warden accept greenhouse p1

Mirrors umweltctl's idiom: --url (UMWELTD_URL), --api-key (UMWELTD_API_KEY), JSON
to stdout, exit 1 on error. The embedded agent resolves lazily and only for
`new`/`warden tick` — every other subcommand works with no SDK installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


def _client(args):
    from umweltd.client import UmweltClient
    return UmweltClient(args.url, api_key=args.api_key)


def _world_client(args, name: str):
    from umweltd.client import UmweltClient
    return UmweltClient(args.url, world=name, api_key=args.api_key)


def _parse_knobs(pairs) -> dict:
    knobs = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"--knob wants key=value, got {pair!r}")
        try:
            knobs[key] = json.loads(value)
        except json.JSONDecodeError:
            knobs[key] = value
    return knobs


def _out(payload) -> None:
    print(json.dumps(payload, indent=1, sort_keys=True))


# ── subcommand bodies ─────────────────────────────────────────────────────────────


def cmd_new(args) -> int:
    if bool(args.rant) == bool(args.file):
        raise ValueError("exactly one of --rant or --file is required")
    rant = args.rant if args.rant else Path(args.file).read_text()

    from umweltforge.agent import resolve_agent
    from umweltforge.pipeline import compile_world
    agent = resolve_agent(args.agent)
    result = compile_world(
        args.name, rant, agent=agent,
        client=None if args.no_register else _client(args),
        root=args.root, max_attempts=args.max_attempts,
        register=not args.no_register, world_knobs=_parse_knobs(args.knob))
    _out(asdict(result))
    return 0 if result.ok else 1


def cmd_validate(args) -> int:
    """Re-run the gate for a forge world (by name) or any 'module:ATTR' ref."""
    if ":" in args.target:
        from umwelt.spec.validate import validate_spec
        report = validate_spec(args.target)
        _out(report.to_dict())
        return 0 if report.ok else 1
    from umweltforge.pipeline import run_validation
    from umweltforge.workspace import ForgeWorkspace
    ws = ForgeWorkspace.open(args.target, root=args.root)
    ok, report = run_validation(ws.root, ws.spec_ref())
    _out(report)
    return 0 if ok else 1


def cmd_warden_tick(args) -> int:
    from umweltforge.agent import resolve_agent
    from umweltforge.warden import warden_tick
    result = warden_tick(args.name, agent=resolve_agent(args.agent),
                         client=_world_client(args, args.name), root=args.root,
                         apply=not args.no_apply)
    _out(asdict(result))
    return 0 if not result.error else 1


def cmd_warden_dial(args) -> int:
    from umweltforge.policy import WardenPolicy, append_ledger
    from umweltforge.workspace import ForgeWorkspace
    ws = ForgeWorkspace.open(args.name, root=args.root)
    policy = WardenPolicy.load(ws.policy_path, args.name)
    getattr(policy, args.dial_action)(args.change_type)   # promote | demote
    policy.save(ws.policy_path)
    append_ledger(ws.ledger_path, {"world": args.name, "action":
                                   args.dial_action + "d",
                                   "change_type": args.change_type, "by": "cli"})
    _out({"world": args.name, "dials": policy.dials})
    return 0


def cmd_warden_verdict(args) -> int:
    from umweltforge.policy import append_ledger
    from umweltforge.workspace import ForgeWorkspace
    if args.verdict == "reject":
        ws = ForgeWorkspace.open(args.name, root=args.root)
        append_ledger(ws.ledger_path, {"world": args.name, "action": "rejected",
                                       "proposal_id": args.proposal_id, "by": "cli"})
        _out({"world": args.name, "proposal": args.proposal_id,
              "verdict": "rejected"})
        return 0
    from umweltforge.warden import apply_accepted
    applied, detail = apply_accepted(args.name, args.proposal_id,
                                     client=_client(args), root=args.root)
    _out({"world": args.name, "proposal": args.proposal_id,
          "applied": applied, **({"detail": detail} if detail else {})})
    return 0 if applied else 1


def cmd_warden_status(args) -> int:
    from umweltforge.policy import (WardenPolicy, competence_summary, read_ledger)
    from umweltforge.workspace import ForgeWorkspace
    ws = ForgeWorkspace.open(args.name, root=args.root)
    policy = WardenPolicy.load(ws.policy_path, args.name)
    entries = read_ledger(ws.ledger_path)
    _out({"world": args.name, "dials": policy.dials,
          "competence": competence_summary(entries),
          "ledger_entries": len(entries)})
    return 0


# ── the parser ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="umwelt-forge",
        description="rant -> authored spec -> deterministic gate -> running world")
    ap.add_argument("--url", default=os.environ.get("UMWELTD_URL",
                                                    "http://127.0.0.1:7071"))
    ap.add_argument("--api-key", default=os.environ.get("UMWELTD_API_KEY"))
    ap.add_argument("--root", default=None,
                    help="forge root dir (default ~/.umwelt/forge or "
                         "UMWELT_FORGE_ROOT)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="author + validate + register a world from a rant")
    p.add_argument("name")
    p.add_argument("--rant", help="the domain description, inline")
    p.add_argument("--file", help="read the domain description from a file")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--agent", default=None, help="agent impl 'module:factory' "
                   "(default: the embedded Claude agent; env UMWELT_FORGE_AGENT)")
    p.add_argument("--no-register", action="store_true",
                   help="author + validate only; skip daemon registration")
    p.add_argument("--knob", action="append", default=[],
                   help="extra world.json knob, key=value (repeatable)")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("validate", help="re-run the gate for a forge world or any "
                                        "'module:ATTR' ref")
    p.add_argument("target")
    p.set_defaults(fn=cmd_validate)

    w = sub.add_parser("warden", help="the inspection intelligence")
    wsub = w.add_subparsers(dest="warden_cmd", required=True)

    p = wsub.add_parser("tick", help="one inspection pass (cron-able)")
    p.add_argument("name")
    p.add_argument("--no-apply", action="store_true",
                   help="propose only, even for change-types dialed to run")
    p.add_argument("--agent", default=None)
    p.set_defaults(fn=cmd_warden_tick)

    for dial in ("promote", "demote"):
        p = wsub.add_parser(dial, help=f"{dial} a change-type "
                            f"{'to' if dial == 'promote' else 'back from'} auto-apply")
        p.add_argument("name")
        p.add_argument("change_type")
        p.set_defaults(fn=cmd_warden_dial, dial_action=dial)

    for verdict in ("accept", "reject"):
        p = wsub.add_parser(verdict, help=f"{verdict} a watched proposal")
        p.add_argument("name")
        p.add_argument("proposal_id")
        p.set_defaults(fn=cmd_warden_verdict, verdict=verdict)

    p = wsub.add_parser("status", help="dials + competence summary")
    p.add_argument("name")
    p.set_defaults(fn=cmd_warden_status)

    return ap


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
