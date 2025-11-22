"""FastAPI application exposing the SocialVision face-enrichment pipeline."""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse

from src.config import get_config
from src.logger import setup_logger
from src.pipeline import FacePipeline

logger = setup_logger(__name__)
config = get_config()
DEFAULT_SIMILARITY_THRESHOLD = getattr(config, "FACE_SIMILARITY_THRESHOLD", 0.35)

app = FastAPI(
    title="SocialVision Face Pipeline",
    description="Face search and continuous enrichment API",
    version="1.0.0",
)

pipeline = FacePipeline()


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


@app.get("/health")
def health_check():
    """Simple readiness probe for container/deployment environments."""
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=getattr(config, "API_PORT", 8080),
        reload=True,
    )
