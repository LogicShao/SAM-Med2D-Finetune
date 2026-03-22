from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web_demo.config import APP_HOST, APP_NAME, APP_PORT, STATIC_DIR, TEMPLATES_DIR, ensure_web_demo_dirs
from web_demo.ui.home import router as home_router
from web_demo.ui.result import router as result_router
from web_demo.ui.run import router as run_router


def create_app() -> FastAPI:
    ensure_web_demo_dirs()
    app = FastAPI(title=APP_NAME)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(home_router)
    app.include_router(run_router)
    app.include_router(result_router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("web_demo.app:app", host=APP_HOST, port=APP_PORT, reload=False)


if __name__ == "__main__":
    main()