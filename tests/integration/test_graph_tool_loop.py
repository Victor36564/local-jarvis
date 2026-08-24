from jarvis.agent_graph import build_graph
from jarvis.config import JarvisConfig


class StubProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        class Dummy:
            def to(self, _):
                return self

        return {"input_ids": Dummy()}

    def batch_decode(self, _ids, skip_special_tokens=True):
        self.calls += 1
        if self.calls == 1:
            return ["{\"tool_name\": \"read_file_content\", \"arguments\": {\"file_path\": \"README.md\"}}"]
        return ["Done"]


class StubModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[1, 2, 3]]


def test_graph_returns_tool_result_path():
    cfg = JarvisConfig()
    graph = build_graph(StubModel(), StubProcessor(), cfg)

    result = graph.invoke(
        {
            "transcript": "read readme",
            "audio": None,
            "screenshot": None,
            "tool_result": None,
            "tool_calls_count": 0,
        },
        config={"recursion_limit": 4},
    )

    assert result.get("final_response") == "Done"
