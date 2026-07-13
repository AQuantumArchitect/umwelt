"""umweltctl — a thin operator CLI over UmweltClient.

Everything here was already reachable by hand-rolling HTTP or scripting
UmweltClient; this just gives the common loop (create a world, check it's alive,
push a batch, read a belief, snapshot) a command instead of a snippet.

    umweltctl worlds
    umweltctl create market --spec umwelt_market.spec:MARKET_SPEC \\
        --vocabulary umwelt_market.vocabulary:register_market_vocabulary
    umweltctl health --world market
    umweltctl ingest --world market --file batch.json
    umweltctl belief --world market --node aapl --role drift
    umweltctl snapshot --world market
    umweltctl stop --world market
    umweltctl start --world market

`--url` defaults to http://127.0.0.1:7071; `--api-key` defaults to the
UMWELTD_API_KEY env var (same as the daemon itself)."""
from __future__ import annotations

import argparse
import json
import os
import sys

from umweltd.client import UmweltClient


def _client(args) -> UmweltClient:
    return UmweltClient(args.url, world=getattr(args, "world", None),
                        api_key=args.api_key or os.environ.get("UMWELTD_API_KEY"))


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_worlds(args) -> None:
    _print(_client(args).list_worlds())


def cmd_create(args) -> None:
    _print(_client(args).create_world(
        args.name, args.spec, vocabulary=args.vocabulary,
        **({"webhook_url": args.webhook_url} if args.webhook_url else {}),
        **({"gauge": args.gauge} if args.gauge else {}),
        **({"pin_rngs": True} if args.pin_rngs else {}),
    ))


def cmd_stop(args) -> None:
    _print(_client(args).stop_world(args.world))


def cmd_start(args) -> None:
    _print(_client(args).start_world(args.world))


def cmd_health(args) -> None:
    _print(_client(args).health())


def cmd_state(args) -> None:
    _print(_client(args).state())


def cmd_belief(args) -> None:
    _print(_client(args).belief(args.node, args.role))


def cmd_recommendations(args) -> None:
    _print(_client(args).recommendations())


def cmd_snapshot(args) -> None:
    _print(_client(args).snapshot())


def cmd_ingest(args) -> None:
    payload = json.loads(open(args.file).read())
    events = payload["events"] if isinstance(payload, dict) else payload
    _print(_client(args).ingest(events, flush_secs=args.flush_secs))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="umweltctl", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("UMWELTD_URL", "http://127.0.0.1:7071"))
    ap.add_argument("--api-key", default=None,
                    help="defaults to the UMWELTD_API_KEY env var")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("worlds", help="list the world catalog").set_defaults(func=cmd_worlds)

    p = sub.add_parser("create", help="create + spawn a new world")
    p.add_argument("name")
    p.add_argument("--spec", required=True, help="'module:ATTR' DomainSpec ref")
    p.add_argument("--vocabulary", default=None, help="'module:fn' vocabulary ref")
    p.add_argument("--webhook-url", default=None)
    p.add_argument("--gauge", default=None, choices=["live", "replay"])
    p.add_argument("--pin-rngs", action="store_true")
    p.set_defaults(func=cmd_create)

    for name, fn, help_ in [
        ("stop", cmd_stop, "stop a world (snapshots on the way out)"),
        ("start", cmd_start, "start (or respawn) a world"),
        ("health", cmd_health, "world health"),
        ("state", cmd_state, "the canonical graph_state projection"),
        ("recommendations", cmd_recommendations, "the shadow decision layer"),
        ("snapshot", cmd_snapshot, "save engine + cursor"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--world", required=True)
        p.set_defaults(func=fn)

    p = sub.add_parser("belief", help="one raw-Bloch belief read")
    p.add_argument("--world", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--role", required=True)
    p.set_defaults(func=cmd_belief)

    p = sub.add_parser("ingest", help="post a batch of events from a JSON file")
    p.add_argument("--world", required=True)
    p.add_argument("--file", required=True,
                   help="JSON: {\"events\": [[ts,sid,value,meta|null],...]} or a bare list")
    p.add_argument("--flush-secs", type=float, default=None)
    p.set_defaults(func=cmd_ingest)

    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"umweltctl: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
