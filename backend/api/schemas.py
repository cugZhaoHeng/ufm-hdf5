"""Pydantic schemas used by the FastAPI endpoints."""
from typing import Literal

from pydantic import BaseModel, Field


class AnimationRequest(BaseModel):
    """Request body for starting a UFM animation task."""

    file_id: str = Field(..., description="Uploaded HDF5 file id returned by /api/files/upload-h5.")
    group_names: list[str] = Field(..., description="Selected Fracture Simulation group names.")
    group_display_names: dict[str, str] | None = Field(
        None,
        description="Optional mapping from Fracture Simulation group name to metadata Name.",
    )
    level: Literal["element", "cell"] = Field(..., description="Animation detail level.")
    property_name: str = Field(..., description="Element or cell property dataset name.")
    renderer: Literal["matplotlib", "pyvista"] = Field("matplotlib", description="Rendering backend.")
    output_format: Literal["gif", "mp4"] = Field("gif", description="Animation output format.")
    well_file_id: str | None = Field(None, description="Optional uploaded well trajectory file id.")
    fps: int = Field(20, ge=1, le=60, description="Frames per second for saved animation.")
    interval: int = Field(50, ge=1, le=5000, description="Matplotlib frame interval in milliseconds.")
    keep_aspect: bool = Field(True, description="Whether to keep the 3D axis aspect ratio.")
    time_step_stride: int = Field(
        1,
        ge=1,
        le=100,
        description="Render one frame every N time steps. The final time step of each group is always kept.",
    )


class FinalStateRequest(BaseModel):
    """Request body for reading final-time UFM data for 3D preview."""

    file_id: str = Field(..., description="Uploaded HDF5 file id returned by /api/files/upload-h5.")
    group_names: list[str] = Field(..., description="Groups to include in the final-state snapshot.")
    level: Literal["element", "cell"] = Field(..., description="Data level used by the preview.")
    property_name: str = Field(..., description="Element or cell property dataset name.")
    well_file_id: str | None = Field(None, description="Optional uploaded well trajectory file id.")


class H5InspectRequest(BaseModel):
    """Request body for parsing an existing HDF5 file."""

    file_id: str = Field(..., description="Existing HDF5 file id returned by /api/files/h5.")


class TaskResponse(BaseModel):
    """Response returned when an animation task is created."""

    task_id: str


class TaskStatus(BaseModel):
    """Current status for a background animation task."""

    task_id: str
    status: str
    percent: int
    current_group: str | None = None
    current_group_index: int = 0
    total_groups: int = 0
    message: str = ""
    result_url: str | None = None
    error: str | None = None
