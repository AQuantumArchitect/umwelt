"""Close the loop: read umweltd's own API telemetry, feed it to the mirror world.

Usage:
    UMWELTD_API_KEY=... python pusher.py [rounds] [sleep_s]

Environment:
    UMWELTD_URL      daemon base URL (default http://127.0.0.1:7071)
    UMWELTD_API_KEY  the daemon's API key (required if one is set)
    MIRROR_WORLD     name of the mirror world (default "mirror")
    WATCHED_WORLD    the sibling world it observes (default "greenhouse")
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib import request

BASE = os.environ.get("UMWELTD_URL", "http://127.0.0.1:7071")
KEY = os.environ.get("UMWELTD_API_KEY", "")
MIRROR = os.environ.get("MIRROR_WORLD", "mirror")
WATCHED = os.environ.get("WATCHED_WORLD", "greenhouse")
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SLEEP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0


def get(path):
    req = request.Request(BASE + path, headers={"X-API-Key": KEY})
    try:
        with request.urlopen(req, timeout=5) as r:
            return json.load(r)
    except Exception:
        return None


def post_events(events):
    body = json.dumps({"events": events}).encode()
    req = request.Request(f"{BASE}/worlds/{MIRROR}/events", data=body, method="POST",
                          headers={"X-API-Key": KEY, "Content-Type": "application/json"})
    with request.urlopen(req, timeout=10) as r:
        return json.load(r)


for i in range(ROUNDS):
    now = datetime.now(timezone.utc).isoformat()
    sup = get("/health")
    worlds = {w["name"]: w for w in (sup or {}).get("worlds", [])}
    gh = get(f"/worlds/{WATCHED}/health")
    me = get(f"/worlds/{MIRROR}/health")

    events = [[now, "sup_pulse", 1.0 if sup and sup.get("ok") else 0.0, None]]
    events.append([now, "gh_alive",
                   1.0 if worlds.get(WATCHED, {}).get("running") else 0.0, None])
    if gh and "events_db_bytes" in gh:
        events.append([now, "gh_bulk", float(gh["events_db_bytes"]), None])
    events.append([now, "self_alive",
                   1.0 if worlds.get(MIRROR, {}).get("running") else 0.0, None])
    if me and "events_db_bytes" in me:
        events.append([now, "self_bulk", float(me["events_db_bytes"]), None])

    res = post_events(events)
    if i % 10 == 0 or i == ROUNDS - 1:
        print(f"round {i:3d}: pushed {len(events)} readings -> {res}", flush=True)
    time.sleep(SLEEP_S)

print("done", flush=True)
