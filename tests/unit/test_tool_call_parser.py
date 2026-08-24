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


def test_parse_tool_call_rejects_invalid_shape():
    text = '{"tool": "web_search", "args": {}}'
    assert parse_tool_call(text) is None
