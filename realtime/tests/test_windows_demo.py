"""Exercise actual HTTP forwarding, response framing and local-origin boundaries."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from contextlib import closing
import http.client
import threading

import pytest

from realtime.scripts.windows_demo import Bridge, private_ip


@pytest.fixture
def bridge():
    class Upstream(BaseHTTPRequestHandler):
        # Buffer the fixture's response so Windows loopback does not split its
        # headers/body across writes. The bridge's output is checked separately.
        wbufsize = 1024 * 1024

        def do_GET(self):
            body = b"x" * 800_000
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert self.headers["X-Session-Key"] == "test-session"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with (
        ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as upstream,
        Bridge(("127.0.0.1", 0), upstream.server_address) as server,
    ):
        threads = [
            threading.Thread(target=item.serve_forever, daemon=True)
            for item in (upstream, server)
        ]
        for thread in threads:
            thread.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            upstream.shutdown()
            for thread in threads:
                thread.join(timeout=2)


def request(address, method="GET", path="/", body=None, headers=None):
    with closing(http.client.HTTPConnection(*address, timeout=2)) as client:
        client.request(
            method,
            path,
            body=body,
            headers={"Host": f"127.0.0.1:{address[1]}", **(headers or {})},
        )
        response = client.getresponse()
        return response.status, response.getheader("Connection"), response.read()


def test_bridge_delivers_complete_assets_and_authenticated_posts(bridge):
    status, connection, body = request(bridge)
    assert (status, connection, len(body)) == (200, "close", 800_000)
    assert (
        request(
            bridge,
            "POST",
            "/api/sessions/test/playback",
            b'{"frame_id":42}',
            {"X-Session-Key": "test-session"},
        )[2]
        == b'{"frame_id":42}'
    )


@pytest.mark.parametrize(
    "path,headers,status",
    [
        ("/", {"Host": "untrusted.example"}, 403),
        ("/", {"Origin": "https://untrusted.example"}, 403),
        ("/", {"Sec-Fetch-Site": "cross-site"}, 403),
        ("http://untrusted.example/", {}, 400),
        ("/api/sessions/test/events", {}, 400),
        ("/", {"Content-Length": "-1"}, 413),
        ("/", {"Transfer-Encoding": "chunked"}, 400),
    ],
)
def test_bridge_rejects_nonlocal_or_unbounded_requests(bridge, path, headers, status):
    assert request(bridge, path=path, headers=headers)[0] == status


def test_private_wsl_address_required():
    assert private_ip("10.77.0.2") == "10.77.0.2"
    for address in ("127.0.0.1", "0.0.0.0", "8.8.8.8", "::1", "not-an-ip"):
        with pytest.raises(ValueError):
            private_ip(address)
