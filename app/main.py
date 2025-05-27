from fastapi import FastAPI, Body, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from app.core.config import settings
from app.schemas.schemas import LocalFileRequest
import logging
import os
import io
from app.services.summarization import process_media_file
from datetime import datetime
from app.db.database import SessionLocal, init_db
from app.models.models import SummarizationRequest

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация базы данных при запуске
@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database initialized")

# Include API routes
app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to Video/Audio Summarization API"}

@app.post("/summarize")
async def summarize_local_file(request_data: LocalFileRequest = Body(...)):
    """Public endpoint to summarize a local video file without authentication."""
    try:
        file_path = request_data.video_path
        logging.info(f"Processing file: {file_path}")
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Открываем файл как UploadFile
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        # Создаем объект UploadFile
        file = UploadFile(
            filename=os.path.basename(file_path),
            file=io.BytesIO(file_content)
        )
        
        # Обрабатываем файл
        result = await process_media_file(file)
        
        # Создаем запись в базе данных с user_id=0
        db = SessionLocal()
        try:
            request = SummarizationRequest(
                user_id=0,
                filename=result["filename"],
                transcript=result["transcript"],
                summary=result["summary"]
            )
            db.add(request)
            db.commit()
            db.refresh(request)
            return request
        finally:
            db.close()
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing your request. Please try again later."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=4, reload=True) 