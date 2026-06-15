"""FastAPI routes for upload, metadata parsing, animation tasks, and downloads."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.schemas import AnimationRequest, FinalStateRequest, H5InspectRequest, TaskResponse, TaskStatus
from backend.services.animation import generate_ufm_animation
from backend.services.files import (
    get_h5_upload_path,
    get_output_path,
    get_well_upload_path,
    list_available_h5_files,
    list_available_well_files,
    make_file_id,
    resolve_file_path,
    save_upload_file,
)
from backend.services.hdf5_inspector import inspect_hdf5_file
from backend.services.snapshot import build_final_state_snapshot
from backend.services.task_manager import TASK_MANAGER
from utils.logger import create_logger

router = APIRouter()
logger = create_logger(__name__)


@router.get("/files/h5", tags=["Files"])
def list_h5_files():
    """List existing HDF5 files available to the current system."""
    return {"files": list_available_h5_files()}


@router.get("/files/well", tags=["Files"])
def list_well_files():
    """List existing well trajectory files available to the current system."""
    return {"files": list_available_well_files()}


@router.post("/files/inspect-h5", tags=["Files"])
def inspect_existing_h5(request: H5InspectRequest):
    """Parse an existing HDF5 file selected by file id."""
    h5_path = resolve_file_path(request.file_id, "h5")
    if h5_path is None:
        raise HTTPException(status_code=404, detail="Selected HDF5 file was not found.")
    try:
        metadata = inspect_hdf5_file(h5_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse HDF5 file: {exc}") from exc

    return {
        "file_id": request.file_id,
        "filename": h5_path.name,
        **metadata,
    }


@router.post("/files/upload-h5", tags=["Files"])
async def upload_h5(file: UploadFile = File(...)):
    """Upload an HDF5 file and return its UFM group/property metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded file name.")
    if Path(file.filename).suffix.lower() not in {".h5", ".hdf5"}:
        raise HTTPException(status_code=400, detail="Only .h5 or .hdf5 files are supported.")

    upload_path = get_h5_upload_path(file.filename)
    await save_upload_file(file, upload_path)
    file_id = make_file_id("upload_h5", upload_path.name)

    try:
        metadata = inspect_hdf5_file(upload_path)
    except Exception as exc:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse HDF5 file: {exc}") from exc

    return {
        "file_id": file_id,
        "filename": file.filename,
        **metadata,
    }


@router.post("/files/upload-well", tags=["Files"])
async def upload_well(file: UploadFile = File(...)):
    """Upload an optional well trajectory text/CSV file for animation overlays."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded file name.")

    upload_path = get_well_upload_path(file.filename)
    await save_upload_file(file, upload_path)
    file_id = make_file_id("upload_well", upload_path.name)
    return {"file_id": file_id, "filename": file.filename}


@router.post("/animations", response_model=TaskResponse, tags=["Animations"])
def create_animation(request: AnimationRequest):
    """Start a background animation generation task."""
    h5_path = resolve_file_path(request.file_id, "h5")
    if h5_path is None:
        raise HTTPException(status_code=404, detail="Uploaded HDF5 file was not found.")

    well_path = resolve_file_path(request.well_file_id, "well") if request.well_file_id else None
    if request.well_file_id and well_path is None:
        raise HTTPException(status_code=404, detail="Uploaded well trajectory file was not found.")

    task_id = TASK_MANAGER.create_task(total_groups=len(request.group_names))

    def job(progress_callback):
        task_start = perf_counter()
        logger.info(
            "Animation task started | task_id=%s renderer=%s level=%s property=%s groups=%s stride=%s format=%s",
            task_id,
            request.renderer,
            request.level,
            request.property_name,
            len(request.group_names),
            request.time_step_stride,
            request.output_format,
        )

        def generate_with_matplotlib():
            return generate_ufm_animation(
                h5_file_path=h5_path,
                group_names=request.group_names,
                group_display_names=request.group_display_names,
                level=request.level,
                property_name=request.property_name,
                output_format=request.output_format,
                output_dir=get_output_path(),
                well_file_path=well_path,
                fps=request.fps,
                interval=request.interval,
                keep_aspect=request.keep_aspect,
                time_step_stride=request.time_step_stride,
                progress_callback=progress_callback,
            )

        try:
            if request.renderer == "pyvista":
                from backend.services.pyvista_animation import generate_pyvista_ufm_animation

                try:
                    result_path = generate_pyvista_ufm_animation(
                        h5_file_path=h5_path,
                        group_names=request.group_names,
                        group_display_names=request.group_display_names,
                        level=request.level,
                        property_name=request.property_name,
                        output_format=request.output_format,
                        output_dir=get_output_path(),
                        well_file_path=well_path,
                        fps=request.fps,
                        keep_aspect=request.keep_aspect,
                        time_step_stride=request.time_step_stride,
                        progress_callback=progress_callback,
                    )
                    elapsed = perf_counter() - task_start
                    logger.info(
                        "Animation task completed | task_id=%s renderer=%s elapsed=%.2fs output=%s",
                        task_id,
                        request.renderer,
                        elapsed,
                        result_path,
                    )
                    return result_path
                except Exception as exc:
                    logger.warning(f"PyVista renderer failed, fallback to Matplotlib: {exc}")
                    progress_callback(
                        percent=1,
                        total_groups=len(request.group_names),
                        message="PyVista renderer failed on this machine. Falling back to Matplotlib.",
                    )
                    result_path = generate_with_matplotlib()
                    elapsed = perf_counter() - task_start
                    logger.info(
                        "Animation task completed after fallback | task_id=%s elapsed=%.2fs output=%s",
                        task_id,
                        elapsed,
                        result_path,
                    )
                    return result_path

            result_path = generate_with_matplotlib()
            elapsed = perf_counter() - task_start
            logger.info(
                "Animation task completed | task_id=%s renderer=%s elapsed=%.2fs output=%s",
                task_id,
                request.renderer,
                elapsed,
                result_path,
            )
            return result_path
        except Exception:
            elapsed = perf_counter() - task_start
            logger.exception("Animation task failed | task_id=%s renderer=%s elapsed=%.2fs", task_id, request.renderer, elapsed)
            raise

    TASK_MANAGER.submit(task_id, job)
    return TaskResponse(task_id=task_id)


@router.post("/snapshots/final-state", tags=["Snapshots"])
def get_final_state_snapshot(request: FinalStateRequest):
    """Return final-time 3D property data for browser-side preview."""
    h5_path = resolve_file_path(request.file_id, "h5")
    if h5_path is None:
        raise HTTPException(status_code=404, detail="Uploaded HDF5 file was not found.")

    well_path = resolve_file_path(request.well_file_id, "well") if request.well_file_id else None
    if request.well_file_id and well_path is None:
        raise HTTPException(status_code=404, detail="Uploaded well trajectory file was not found.")

    try:
        return build_final_state_snapshot(
            h5_file_path=h5_path,
            group_names=request.group_names,
            level=request.level,
            property_name=request.property_name,
            well_file_path=well_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build final-state snapshot: {exc}") from exc


@router.get("/tasks/{task_id}", response_model=TaskStatus, tags=["Tasks"])
def get_task(task_id: str):
    """Return current progress for an animation generation task."""
    task = TASK_MANAGER.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")
    return TaskStatus(**task)


@router.get("/tasks/{task_id}/result", tags=["Tasks"])
def download_result(task_id: str):
    """Download the generated animation file for a finished task."""
    task = TASK_MANAGER.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task was not found.")
    if task["status"] != "completed" or not task.get("result_path"):
        raise HTTPException(status_code=409, detail="Task has not completed yet.")

    result_path = Path(task["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Generated file was not found.")
    return FileResponse(result_path, filename=result_path.name)
