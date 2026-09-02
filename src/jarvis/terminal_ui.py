from __future__ import annotations

# Status-only console output; application logs stay in the parent terminal.
import os
import socket
import sys
from argparse import ArgumentParser
from typing import TextIO


class TerminalUI:
    """Small, flushed status display for the Jarvis companion console."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stdout
        self._green = os.name == "nt" and getattr(self.stream, "isatty", lambda: False)()

    def status(self, message: str) -> None:
        # Flush each event so startup and interaction states appear immediately.
        prefix = "\033[92m" if self._green else ""
        suffix = "\033[0m" if self._green else ""
        self.stream.write(f"{prefix}{message}{suffix}\n")
        self.stream.flush()

    def startup(self, message: str) -> None:
        self.status(message)

    def listening(self) -> None:
        self.status("Listening...")

    def wake_triggered(self) -> None:
        self.status("Wake word triggered. Retrieving audio...")

    def tool(self, message: str) -> None:
        self.status(message)

    def response(self, message: str) -> None:
        self.status(f"Final response: {message}")


class SocketTerminalUI(TerminalUI):
    """Status sink that forwards messages to the dedicated popup console."""

    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self.stream = None
        self._green = False

    def status(self, message: str) -> None:
        try:
            self.connection.sendall(f"{message}\n".encode())
        except OSError:
            pass

    def close(self) -> None:
        self.connection.close()


def run_popup(port: int) -> None:
    # The popup receives newline-delimited status messages from the parent.
    with socket.create_connection(("127.0.0.1", port)) as connection:
        terminal = TerminalUI()
        with connection.makefile("r", encoding="utf-8") as messages:
            for message in messages:
                terminal.status(message.rstrip("\n"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Jarvis status console")
    parser.add_argument("--port", type=int, required=True)
    run_popup(parser.parse_args().port)