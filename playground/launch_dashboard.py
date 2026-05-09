#!/usr/bin/env python3
"""Serve the teutonic.ai-style dashboard locally against the devnet's minio.

The validator already uploads `dashboard.json` and `index.html` to its
configured R2 bucket via `R2.put_dashboard*` (see validator.py:1726 and
2495). On production that's Cloudflare R2 served behind teutonic.ai;
here it's our local minio. The frontend (website/index.html) hardcodes
DATA_ENDPOINTS / HTML_ENDPOINTS pointing at Hippius URLs — we patch
those in memory at serve-time so we don't have to modify the upstream
HTML file.

Usage:
    docker compose -f playground/docker-compose.yml up -d   # storage + chain
    source playground/env.devnet.sh
    python -m playground.launch_validator &        # produces dashboard.json
    python -m playground.launch_dashboard          # serves it on :9300

    # then open http://localhost:9300/

The server polls minio for fresh dashboard.json on every page load (no
caching). Auto-version-update in the HTML stays disabled because the
HTML_ENDPOINTS check runs against minio and would compare the file
against itself.
"""
from __future__ import annotations

import http.server
import logging
import os
import re
import sys
from pathlib import Path
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("dashboard")

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
WEBSITE_HTML = PLAYGROUND_ROOT / "website" / "index.html"

LISTEN_HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("DASHBOARD_PORT", "9300"))


def _patch_html(raw: str, *, dashboard_url: str, index_url: str) -> str:
    """Replace DATA_ENDPOINTS and HTML_ENDPOINTS with single local URLs.

    The originals are JS array literals like
        var DATA_ENDPOINTS = [
            "https://...hippius.com/.../dashboard.json", ...
        ];
    The frontend iterates these in race-fashion. Reducing each to a
    one-element array pointing at our proxy keeps the iteration logic
    happy without any code changes.
    """
    raw = re.sub(
        r"var DATA_ENDPOINTS = \[[^\]]*\];",
        f'var DATA_ENDPOINTS = ["{dashboard_url}"];',
        raw,
    )
    raw = re.sub(
        r"var HTML_ENDPOINTS = \[[^\]]*\];",
        f'var HTML_ENDPOINTS = ["{index_url}"];',
        raw,
    )
    return raw


def _build_minio_url(key: str) -> str:
    """Resolve a key in our minio bucket to a presigned-or-public URL.

    minio's default 'private' policy 403s anonymous reads. We sign a
    long-lived URL on every request — overhead is negligible and the
    URL is only handed to the localhost browser.
    """
    import boto3
    from botocore.client import Config as BotoConfig
    c = boto3.client(
        "s3",
        endpoint_url=os.environ["TEUTONIC_R2_ENDPOINT"],
        aws_access_key_id=os.environ["TEUTONIC_R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TEUTONIC_R2_SECRET_KEY"],
        region_name="auto",
        config=BotoConfig(signature_version="s3v4",
                          s3={"addressing_style": "path"}),
    )
    return c.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["TEUTONIC_R2_BUCKET"], "Key": key},
        ExpiresIn=3600,
    )


class Handler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path.startswith("/dashboard.json"):
            self._proxy_minio("dashboard.json", "application/json")
        elif self.path.startswith("/favicon"):
            ext = self.path.split(".")[-1]
            self._proxy_local(WEBSITE_HTML.parent / Path(self.path).name,
                              f"image/{ext}")
        else:
            self.send_error(404)

    def _serve_html(self):
        raw = WEBSITE_HTML.read_text(encoding="utf-8")
        patched = _patch_html(
            raw,
            dashboard_url="/dashboard.json",
            index_url="/index.html",
        )
        body = patched.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy_minio(self, key: str, content_type: str):
        url = _build_minio_url(key)
        try:
            with urlopen(url, timeout=10) as r:
                body = r.read()
        except Exception as e:
            self.send_error(502, f"minio {key}: {e}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _proxy_local(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)


def main() -> int:
    if not WEBSITE_HTML.exists():
        raise SystemExit(f"website/index.html not found at {WEBSITE_HTML}")
    for v in ("TEUTONIC_R2_ENDPOINT", "TEUTONIC_R2_ACCESS_KEY",
              "TEUTONIC_R2_SECRET_KEY", "TEUTONIC_R2_BUCKET"):
        if not os.environ.get(v):
            raise SystemExit(
                f"{v} unset; run `source playground/env.devnet.sh` first"
            )

    addr = (LISTEN_HOST, LISTEN_PORT)
    server = http.server.ThreadingHTTPServer(addr, Handler)
    log.info("dashboard serving at http://%s:%d/  (Ctrl+C to stop)",
             LISTEN_HOST, LISTEN_PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
