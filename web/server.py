"""
Lightweight Web Server for Sudoku Reader & Solver Web UI.
Uses standard Python library (no extra dependencies required).
Launches local HTTP server and automatically opens the browser.
"""

import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8000
WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomHandler(SimpleHTTPRequestHandler):
    """Serves files from the web directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[Web UI] {format % args}\n")


def run_server(port: int = PORT):
    """Starts HTTP server on specified port and opens default browser."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, CustomHandler)
    url = f"http://localhost:{port}/index.html"

    print("=" * 65)
    print("   SUDOKU READER & SOLVER WEB DASHBOARD")
    print("=" * 65)
    print(f"Serving at:  {url}")
    print("Opening your default web browser automatically...")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 65)

    # Automatically open web browser
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server. Done.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
