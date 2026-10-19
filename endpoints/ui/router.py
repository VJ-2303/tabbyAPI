"""Serves the built-in chat client as static files."""

import pathlib

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

STATIC_DIR = pathlib.Path(__file__).parent / "static"

# The client keeps an API key in local storage, so it gets a strict policy:
# only its own scripts and styles, no framing. Requests may go to any server,
# since the client can be pointed at another TabbyAPI instance. Images can be
# data: URLs (SVG previews) and blob: URLs (attached images), never remote.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src *; img-src 'self' data: blob:; "
    "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
)


class ClientFiles(StaticFiles):
    """Static files with headers suited to the chat client."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)

        # Revalidate on every load (a cheap 304 when unchanged), so an updated
        # client is picked up right away instead of after a heuristic expiry
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


def mount_client(app: FastAPI, path: str, directory: pathlib.Path):
    """
    Serve a static client under the given path, at `path/` itself.

    The files need no auth: they hold no secrets, and the client asks for an
    API key before it calls anything.
    """

    @app.get(path, include_in_schema=False)
    async def client_redirect():
        # Relative asset URLs in the page need the trailing slash
        return RedirectResponse(f"{path}/")

    app.mount(path, ClientFiles(directory=directory, html=True), name=path.strip("/"))


def mount_ui(app: FastAPI, path: str = "/ui"):
    """Mount the chat client, plain HTML, CSS and JavaScript talking to the API."""

    mount_client(app, path, STATIC_DIR)


def mount_stats(app: FastAPI, path: str = "/stats"):
    """Mount the statistics page."""

    mount_client(app, path, STATIC_DIR / "stats")
