from html.parser import HTMLParser
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import create_app
from config import Settings
from gpu_services import create_gpu_app


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and "src" in attrs:
            self.urls.append(attrs["src"])
        elif tag == "link" and "href" in attrs:
            self.urls.append(attrs["href"])


@pytest.mark.parametrize("service", ["api", "diarization"])
def test_docs_assets_are_available_without_cdn(service, tmp_path):
    settings = Settings(job_directory=str(tmp_path))
    app = (
        create_app(SimpleNamespace(), settings)
        if service == "api"
        else create_gpu_app(SimpleNamespace(load=lambda: None), settings)
    )
    with TestClient(app) as client:
        response = client.get("/docs")
        assert response.status_code == 200
        assert '"validatorUrl": null' in response.text
        resources = Resources()
        resources.feed(response.text)
        assert len(resources.urls) == 3
        for url in resources.urls:
            assert url.startswith("/docs-assets/")
            asset = client.get(url)
            assert asset.status_code == 200 and len(asset.content) > 100
        assert client.get("/openapi.json").status_code == 200
