from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import Config
from api.routes import random_forest_router, knn_router
from utils.schemas import HealthCheckResponse


# Khởi tạo FastAPI app
app = FastAPI(
    title=Config.PROJECT_NAME,
    description=Config.DESCRIPTION,
    version=Config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    #Tạm thời cho mọi nguồn
    #allow_origins=Config.ALLOWED_ORIGINS,
    allow_origins=["*"],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/", response_model=HealthCheckResponse, tags=["Health Check"])
async def health_check():
    return {
        "status": "healthy",
        "message": f"{Config.PROJECT_NAME} v{Config.VERSION} đang hoạt động",
        "models": {
            "random_forest": "available - Hypertension Classification",
            "knn_systolic": "available - Systolic BP Prediction (R²≈0.42)",
            "knn_diastolic": "available - Diastolic BP Prediction (R²≈0.38)"
        }
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health Check"])
async def health():
    return await health_check()


# Include routers
app.include_router(random_forest_router, prefix=Config.API_V1_STR)
app.include_router(knn_router, prefix=Config.API_V1_STR)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD
    )