from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from sam_med2d_finetune.web_demo.services.results import get_viewer_file_for_result, load_result_view
from sam_med2d_finetune.web_demo.ui.common import build_page_context


router = APIRouter()


@router.get("/results/{result_id}", response_class=HTMLResponse)
def result_page(request: Request, result_id: str) -> HTMLResponse:
    mode_key = request.query_params.get("mode")
    try:
        result = load_result_view(result_id, mode_key=mode_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return request.app.state.templates.TemplateResponse(
        request=request,
        name="result.html",
        context=build_page_context(
            request,
            title=str(result["case_id"]),
            active_nav="samples",
            mode_key=str(result["mode"]["key"]),
            result=result,
        ),
    )


@router.get("/viewer/{result_id}")
def viewer_file(request: Request, result_id: str) -> FileResponse:
    mode_key = request.query_params.get("mode")
    mask_name = request.query_params.get("mask")
    try:
        viewer_path = get_viewer_file_for_result(result_id, mode_key=mode_key, mask_name=mask_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(str(viewer_path), media_type="text/html")
