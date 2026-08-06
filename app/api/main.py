"""FastAPI entry point.

Development command:

    uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.questions import router as questions_router
from app.api.routes.search import router as search_router


def create_app() -> FastAPI:
    application = FastAPI(
        title="Medical Document Assistant",
        version="1.0.0-day14",
        description=(
            "Synthetic medical-document ingestion and citation-grounded "
            "RAG demonstration API."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    application.include_router(health_router)
    application.include_router(documents_router)
    application.include_router(search_router)
    application.include_router(questions_router)

    @application.get("/", tags=["service"])
    def root() -> dict[str, str]:
        return {
            "service": "Medical Document Assistant",
            "health": "/health",
            "documentation": "/docs",
        }

    return application


app = create_app()
