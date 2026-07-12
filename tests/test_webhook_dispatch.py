"""The webhook dispatcher: an AUTO Action POSTs to the app's sink as JSON; a dead
sink never kills the world. (Shadow stays the law — this is only where flipped
outputs go; the engine-side gating is pinned by test_egress_tendrils.)"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from umwelt.membranes.egress import Action
from umweltd.worker import _make_webhook_dispatch


def test_webhook_posts_the_action_as_json():
    received: list[dict] = []

    class Sink(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            received.append(json.loads(self.rfile.read(n)))
            self.send_response(200)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Sink)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/actions"
        dispatch = _make_webhook_dispatch(url)
        dispatch(Action(actuator_id="dev_1", command={"level": 0.7}, node="cell",
                        role="t_level", value=1.0, confidence=0.9, reason="x_auto"))
        assert len(received) == 1
        assert received[0]["actuator_id"] == "dev_1"
        assert received[0]["command"] == {"level": 0.7}
    finally:
        server.shutdown()


def test_dead_sink_is_survivable():
    dispatch = _make_webhook_dispatch("http://127.0.0.1:1/nothing-listens-here")
    dispatch(Action(actuator_id="dev_1", command={}, node="n", role="r",
                    value=1.0, confidence=1.0, reason="x_auto"))   # must not raise