"""FastAPI application exposing the SocialVision face-enrichment pipeline."""

from pathlib import Path
import base64
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from src.config import get_config
from src.logger import setup_logger
from src.pipeline import FacePipeline
from src.image_utils import ImageProcessor
from src.search_engine import SearchEngine
from src.database import FaceDatabase
from src.face_quality import AutoFaceImprover, FaceQualityAssessor

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)
DEFAULT_FRAME_STRIDE = getattr(config, "VIDEO_FRAME_STRIDE", 5)
DEFAULT_MAX_FRAMES = getattr(config, "VIDEO_MAX_FRAMES", 90)

app = FastAPI(
    title="SocialVision Face Pipeline",
    description="Face search and continuous enrichment API with live camera and video support",
    version="2.0.0",
)

pipeline = FacePipeline()
db = FaceDatabase()
search_engine = SearchEngine(db)
auto_improver = AutoFaceImprover(pipeline.search_engine.face_engine, db)


class CameraFrameRequest(BaseModel):
    """Request model for camera frame processing."""
    image_data: str  # Base64 encoded image
    source: Optional[str] = "camera"
    threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD
    top_k: Optional[int] = 10
    auto_improve: Optional[bool] = True


@app.post("/api/enrich-face")
async def enrich_face(
    image: UploadFile = File(..., description="Image containing a face"),
    source: str = Form("unknown"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(10),
):
    """REST endpoint for the enrichment workflow."""
    try:
        content = await image.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles streaming
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to read uploaded file")

    result = pipeline.process_image_bytes(
        image_bytes=content,
        source=source,
        threshold=threshold,
        top_k=top_k,
    )

    if not result.is_match:
        return PlainTextResponse("Person not found.", status_code=404)

    return JSONResponse(result.response)


@app.post("/api/enrich-video")
async def enrich_video(
    video: UploadFile = File(..., description="Video containing faces"),
    source: str = Form("unknown"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(10),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    max_frames: int = Form(DEFAULT_MAX_FRAMES),
):
    """Run the enrichment pipeline over sampled frames from an uploaded video."""

    try:
        content = await video.read()
    except Exception as exc:  # pragma: no cover - FastAPI handles streaming
        logger.error("Failed to read uploaded video: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to read uploaded video")

    result = pipeline.process_video_bytes(
        video_bytes=content,
        source=source,
        threshold=threshold,
        top_k=top_k,
        frame_stride=frame_stride,
        max_frames=max_frames,
        suffix=Path(video.filename or "upload.mp4").suffix or ".mp4",
    )

    if not result.is_match:
        return PlainTextResponse("Person not found in video.", status_code=404)

    return JSONResponse(result.response)


@app.post("/api/search-face")
async def search_face(
    image: UploadFile = File(..., description="Image containing a face to search"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(50),
):
    """Search for similar faces in the database."""
    try:
        content = await image.read()
        image_array = ImageProcessor.load_image_from_bytes(content)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        results = search_engine.search_by_image(
            image=image_array,
            threshold=threshold,
            top_k=top_k,
        )
        return JSONResponse(results)
    except Exception as exc:
        logger.error(f"Error in search_face: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search-video")
async def search_video(
    video: UploadFile = File(..., description="Video containing faces to search"),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(50),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    max_frames: int = Form(DEFAULT_MAX_FRAMES),
):
    """Search for faces in a video."""
    try:
        content = await video.read()
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        
        from src.image_utils import VideoProcessor
        frames = VideoProcessor.sample_frames_from_bytes(
            content,
            suffix=suffix,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")
        
        results = search_engine.search_video_frames(
            frames=frames,
            threshold=threshold,
            top_k=top_k,
        )
        return JSONResponse(results)
    except Exception as exc:
        logger.error(f"Error in search_video: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/search-camera")
async def search_camera(request: CameraFrameRequest = Body(...)):
    """Search for faces from a camera frame (base64 encoded image)."""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        image_array = ImageProcessor.load_image_from_bytes(image_bytes)
        
        if image_array is None:
            raise HTTPException(status_code=400, detail="Unable to decode camera frame")

        results = search_engine.search_by_image(
            image=image_array,
            threshold=request.threshold or DEFAULT_SIMILARITY_THRESHOLD,
            top_k=request.top_k or 50,
        )
        return JSONResponse(results)
    except Exception as exc:
        logger.error(f"Error in search_camera: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/add-face")
async def add_face(
    image: UploadFile = File(..., description="Image containing faces to add"),
    username: str = Form(..., description="Username identifier"),
    source: str = Form("profile_pic"),
    auto_improve: bool = Form(True),
    min_quality_score: float = Form(0.3),
):
    """Add faces to the database with optional auto-improvement."""
    try:
        content = await image.read()
        image_array = ImageProcessor.load_image_from_bytes(content)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")

        if auto_improve:
            summary = auto_improver.improve_and_add_face(
                username=username,
                image=image_array,
                source=source,
                min_quality_score=min_quality_score,
                auto_improve=True,
            )
        else:
            from src.face_recognition_engine import FaceRecognitionEngine
            face_engine = FaceRecognitionEngine(database=db)
            summary = face_engine.process_and_add_face(
                username=username,
                image=image_array,
                source=source,
                return_summary=True,
            )

        if summary.get("success"):
            return JSONResponse({
                "success": True,
                "faces_added": summary.get("faces_added", 0),
                "faces_detected": summary.get("faces_detected", 0),
                "quality_scores": summary.get("quality_scores", []),
            })
        else:
            return JSONResponse({
                "success": False,
                "errors": summary.get("errors", []),
            }, status_code=400)
    except Exception as exc:
        logger.error(f"Error in add_face: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/add-video")
async def add_video(
    video: UploadFile = File(..., description="Video containing faces to add"),
    username: str = Form(..., description="Username identifier"),
    source: str = Form("video"),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    max_frames: int = Form(DEFAULT_MAX_FRAMES),
    auto_improve: bool = Form(True),
    min_quality_score: float = Form(0.3),
):
    """Add faces from video frames to the database."""
    try:
        content = await video.read()
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        
        from src.image_utils import VideoProcessor
        frames = VideoProcessor.sample_frames_from_bytes(
            content,
            suffix=suffix,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from video")

        total_added = 0
        total_detected = 0
        quality_scores = []
        errors = []

        from src.face_recognition_engine import FaceRecognitionEngine
        face_engine = FaceRecognitionEngine(database=db)

        for frame_idx, frame in enumerate(frames):
            if auto_improve:
                summary = auto_improver.improve_and_add_face(
                    username=username,
                    image=frame,
                    source=source,
                    min_quality_score=min_quality_score,
                    auto_improve=True,
                    metadata={"frame_index": frame_idx, "origin": "api_video"},
                )
            else:
                summary = face_engine.process_and_add_face(
                    username=username,
                    image=frame,
                    source=source,
                    metadata={"frame_index": frame_idx, "origin": "api_video"},
                    return_summary=True,
                )
            
            total_added += summary.get("faces_added", 0)
            total_detected += summary.get("faces_detected", 0)
            quality_scores.extend(summary.get("quality_scores", []))
            errors.extend(summary.get("errors", []))

        return JSONResponse({
            "success": total_added > 0,
            "faces_added": total_added,
            "faces_detected": total_detected,
            "frames_processed": len(frames),
            "quality_scores": quality_scores,
            "errors": errors,
        })
    except Exception as exc:
        logger.error(f"Error in add_video: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health_check():
    """Simple readiness probe for container/deployment environments."""
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=getattr(config, "API_PORT", 8080),
        reload=True,
    )
