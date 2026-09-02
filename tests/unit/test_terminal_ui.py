from io import StringIO

from jarvis import main
from jarvis.terminal_ui import SocketTerminalUI, TerminalUI


class TtyBuffer(StringIO):
    def isatty(self):
        return True


def test_terminal_status_is_green_and_flushed(monkeypatch):
    stream = TtyBuffer()
    monkeypatch.setattr("jarvis.terminal_ui.os.name", "nt")
    terminal = TerminalUI(stream)

    terminal.listening()

    assert stream.getvalue() == "\033[92mListening...\033[0m\n"


def test_windows_launcher_spawns_status_popup(monkeypatch):
    calls = []
    monkeypatch.setattr(main.os, "name", "nt")
    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        port = int(command[-1])
        client = main.socket.create_connection(("127.0.0.1", port))
        client.settimeout(1)
        return client

    monkeypatch.setattr(main.subprocess, "Popen", fake_popen)
    terminal = main._open_status_terminal()

    command, kwargs = calls[0]
    assert command[:3] == [main.sys.executable, "-m", "jarvis.terminal_ui"]
    assert kwargs["creationflags"] == main.subprocess.CREATE_NEW_CONSOLE
    assert isinstance(terminal, SocketTerminalUI)
    terminal.close()


def test_non_windows_launcher_uses_current_terminal(monkeypatch):
    monkeypatch.setattr(main.os, "name", "posix")

    assert type(main._open_status_terminal()) is TerminalUI