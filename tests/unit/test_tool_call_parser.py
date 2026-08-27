from jarvis.tool_call_parser import parse_tool_call


def test_parse_tool_call_from_clean_json():
    text = '{"tool_name": "web_search", "arguments": {"query": "cmu", "open_in_browser": false}}'
    call = parse_tool_call(text)

    assert call is not None
    assert call.tool_name == "web_search"
    assert call.arguments["query"] == "cmu"


def test_parse_tool_call_from_wrapped_text():
    text = "Reasoning... {\"tool_name\": \"read_file_content\", \"arguments\": {\"file_path\": \"README.md\"}}"
    call = parse_tool_call(text)

    assert call is not None
    assert call.tool_name == "read_file_content"


def test_parse_tool_call_from_yaml_like_model_output():
    text = '''Here is the proposed action:
tool_name: terminal
arguments:
  command: echo "Hi, my name is Victor" > ~/Desktop/test.txt'''

    call = parse_tool_call(text)

    assert call is not None
    assert call.tool_name == "terminal"
    assert call.arguments["command"].startswith("echo")


def test_parse_tool_call_when_prompt_contains_other_json_objects():
    text = (
        'Example: {"tool_name": "create_note", "arguments": {"content": "x"}}\n'
        'Generated: {"tool_name": "execute_terminal_command", '
        '"arguments": {"command": "nvidia-smi"}}'
    )

    call = parse_tool_call(text)

    assert call is not None
    assert call.tool_name == "execute_terminal_command"


def test_parse_tool_call_rejects_invalid_shape():
    text = '{"tool": "web_search", "args": {}}'
    assert parse_tool_call(text) is None
