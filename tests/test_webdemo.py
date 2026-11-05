"""webdemo serve: only web/ and the model files are reachable."""

import json
import urllib.error
import urllib.request

import pytest

from geoguessr_ai import webdemo


@pytest.fixture
def server(tmp_path):
    model, onnx_dir = tmp_path / "model", tmp_path / "onnx"
    model.mkdir()
    onnx_dir.mkdir()
    (model / "classes.json").write_text(json.dumps(["DE"]))
    (model / "model.safetensors").write_bytes(b"secret weights")
    (onnx_dir / "model_int8.onnx").write_bytes(b"onnx")
    srv = webdemo.start_server(model, onnx_dir, 0)
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def test_serves_page_and_model_files(server):
    status, body = get(f"{server}/web/index.html")
    assert status == 200 and b"onnxruntime-web" in body
    assert get(f"{server}/web/preprocess.js")[0] == 200
    assert json.loads(get(f"{server}/model/classes.json?x=1")[1]) == ["DE"]
    assert get(f"{server}/model/onnx/model_int8.onnx")[1] == b"onnx"


@pytest.mark.parametrize("path", ["/model/model.safetensors", "/pyproject.toml",
                                  "/web/../pyproject.toml", "/model/onnx/../classes.json"])
def test_nothing_else(server, path):
    with pytest.raises(urllib.error.HTTPError) as e:
        get(server + path)
    assert e.value.code == 404


def test_page_links_and_notes():
    html = (webdemo.WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "https://github.com/Daggerabuse/geoguessr-ai" in html
    assert "never leaves your device" in html and "Limited to" in html
    readme = (webdemo.WEB_DIR / "README.md").read_text(encoding="utf-8")
    assert "sdk: static" in readme and "app_file: index.html" in readme
