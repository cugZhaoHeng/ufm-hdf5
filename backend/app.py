"""FastAPI application entrypoint for the UFM HDF5 visualization service."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router

PROJECT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_DIR / "frontend"
OUTPUT_DIR = PROJECT_DIR / "backend" / "storage" / "outputs"

OPENAPI_TAGS = [
    {
        "name": "Files",
        "description": "Upload HDF5 files and optional well trajectory files.",
    },
    {
        "name": "Animations",
        "description": "Create UFM element/cell animation generation tasks.",
    },
    {
        "name": "Snapshots",
        "description": "Read final-time UFM property data for browser-side 3D preview.",
    },
    {
        "name": "Tasks",
        "description": "Query animation task progress and download generated results.",
    },
]

app = FastAPI(
    title="UFM HDF5 Visualization API",
    description="Parse UFM HDF5 files and generate element/cell level fracture animations.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")


@app.get("/", include_in_schema=False)
def index():
    """Serve the EasyUI frontend page."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{asset_name}", include_in_schema=False)
def frontend_asset(asset_name: str):
    """Serve root-level frontend assets used by direct and Nginx deployments."""
    allowed_assets = {"app.js", "config.js", "styles.css", "stride.html"}
    if asset_name not in allowed_assets:
        return FileResponse(FRONTEND_DIR / "index.html")
    return FileResponse(FRONTEND_DIR / asset_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8090, reload=False)
