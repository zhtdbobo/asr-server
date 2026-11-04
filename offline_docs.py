"""Serve the API documentation with assets bundled in the Python package."""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

import swagger_assets


def install_offline_docs(app: FastAPI):
    app.mount(
        "/docs-assets",
        StaticFiles(directory=Path(swagger_assets.__file__).parent),
        name="docs-assets",
    )

    @app.get("/docs", include_in_schema=False)
    def docs(request: Request):
        prefix = request.scope.get("root_path", "").rstrip("/")
        return get_swagger_ui_html(
            openapi_url=prefix + app.openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url=f"{prefix}/docs-assets/swagger-ui-bundle.js",
            swagger_css_url=f"{prefix}/docs-assets/swagger-ui.css",
            swagger_favicon_url=f"{prefix}/docs-assets/favicon-32x32.png",
            swagger_ui_parameters={"validatorUrl": None},
        )
