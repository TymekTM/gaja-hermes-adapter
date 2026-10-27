#!/usr/bin/env python3
"""Authorize Gaja's own Spotify account via OAuth2 Authorization Code + PKCE.

    uv run python scripts/spotify_login.py --client-id <CLIENT_ID> --redirect-uri <REDIRECT_URI>

Prints the authorize URL, reads the final redirected URL pasted on stdin (so the browser can run on
another machine), exchanges the authorization code for tokens, and stores
``{access_token, refresh_token, expires_at, scope, client_id}`` in ``data/spotify_token.json`` with
mode 0600. Never reuses Hermes's Spotify credentials.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import secrets
import sys
import time
import urllib.parse
from pathlib import Path
from typing import TextIO

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaja.config import Settings, resolve  # noqa: E402
from gaja.tools.spotify_tool import (  # noqa: E402
    SPOTIFY_AUTH_URL,
    SPOTIFY_SCOPES,
    SPOTIFY_TOKEN_URL,
    write_token_atomic,
)


def make_pkce_pair() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for RFC 7636 S256 PKCE."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    scope: str = SPOTIFY_SCOPES,
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": state,
        "scope": scope,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    }
    return f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(params)}"


def parse_redirect_input(pasted: str, expected_state: str) -> str:
    """Extract authorization ``code`` from the pasted redirect URL and require matching ``state``."""
    raw = pasted.strip()
    if not raw:
        raise ValueError("empty redirect URL")
    parsed = urllib.parse.urlparse(raw)
    qs = urllib.parse.parse_qs(parsed.query)
    if not qs and "code=" in raw and "?" not in raw:
        qs = urllib.parse.parse_qs(raw)
    if "error" in qs:
        raise ValueError(f"Spotify authorization returned error: {qs['error'][0]}")
    states = qs.get("state")
    if not states or not states[0].strip():
        raise ValueError("missing 'state' query parameter in redirect URL")
    if states[0] != expected_state:
        raise ValueError("OAuth state mismatch in redirect URL")
    codes = qs.get("code")
    if not codes or not codes[0].strip():
        raise ValueError("no 'code' query parameter found in pasted redirect URL")
    return codes[0].strip()


def run_login(
    *,
    client_id: str,
    redirect_uri: str,
    token_path: Path,
    stdin: TextIO = sys.stdin,
    stdout: TextIO = sys.stdout,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    verifier, challenge = make_pkce_pair()
    state = secrets.token_urlsafe(16)
    auth_url = build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        code_challenge=challenge,
        state=state,
    )
    print("Open this URL in your browser and authorize Gaja:\n", file=stdout)
    print(auth_url, file=stdout)
    print("\nPaste the full redirected URL here and press Enter: ", end="", file=stdout, flush=True)
    pasted = stdin.readline()
    code = parse_redirect_input(pasted, state)

    with httpx.Client(timeout=15.0, transport=transport) as client:
        resp = client.post(
            SPOTIFY_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Spotify token exchange failed with HTTP {resp.status_code}")

    body = resp.json()
    access_token = str(body.get("access_token") or "").strip()
    refresh_token = str(body.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        raise RuntimeError("Spotify token response missing access_token or refresh_token")
    expires_at = int(time.time()) + int(body.get("expires_in") or 3600)
    scope = str(body.get("scope") or SPOTIFY_SCOPES)

    payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "scope": scope,
        "client_id": client_id,
    }
    write_token_atomic(token_path, payload)
    print(f"Saved Spotify token to {token_path} (mode 0600).", file=stdout)
    return token_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--client-id", required=True, help="Spotify application Client ID for Gaja")
    parser.add_argument("--redirect-uri", required=True, help="Redirect URI registered on the Spotify app")
    parser.add_argument(
        "--token-path",
        type=Path,
        default=None,
        help="Destination token file (default: [spotify] token_path = data/spotify_token.json)",
    )
    args = parser.parse_args(argv)
    target = resolve((args.token_path or Settings().spotify.token_path).expanduser())
    try:
        run_login(
            client_id=args.client_id.strip(),
            redirect_uri=args.redirect_uri.strip(),
            token_path=target,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
