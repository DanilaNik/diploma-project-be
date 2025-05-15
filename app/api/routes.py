from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.config import settings
from app.db.database import get_db
from app.models.models import User, SummarizationRequest
from app.schemas.schemas import UserCreate, UserResponse, Token, SummarizationRequestResponse
from app.services.summarization import process_media_file, validate_file, translate_text
from app.core.security import get_password_hash, verify_password, get_current_user
from app.crud import get_user_by_email, create_user, authenticate_user
from app.auth import create_access_token

logger = logging.getLogger(__name__)
router = APIRouter()

# Схема аутентификации для защищенных ручек
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Схема аутентификации для суммаризации (опциональная)
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    try:
        # Check if user already exists
        db_user = get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user.password)
        db_user = User(
            email=user.email,
            name=user.name,
            hashed_password=hashed_password
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error registering user"
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return JWT token."""
    try:
        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error logging in user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error logging in user"
        )

@router.post("/summarize")
async def create_summarization(
    file: UploadFile,
    language: str = Form(...),
    token: str = Depends(optional_oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        # Определяем user_id в зависимости от наличия токена
        user_id = None  # По умолчанию для неавторизованных пользователей
        if token:
            try:
                current_user = await get_current_user(token, db)
                user_id = current_user.id
            except Exception as e:
                logger.warning(f"Invalid token: {str(e)}")
                # Продолжаем выполнение без user_id
        
        # Обрабатываем файл
        result = await process_media_file(file, language)
        
        # Создаем запись в базе данных
        db_request = SummarizationRequest(
            user_id=user_id,  # Может быть None для неавторизованных пользователей
            filename=file.filename,  # Используем оригинальное имя файла
            transcript=result["transcription"],  # Используем правильный ключ
            summary=result["summary"]
        )
        db.add(db_request)
        db.commit()
        db.refresh(db_request)
        
        response = {
            "id": db_request.id,
            "filename": db_request.filename,
            "transcript": db_request.transcript,
            "summary": db_request.summary,
            "created_at": db_request.created_at
        }
        
        # Добавляем перевод, если он есть
        if result.get("translation"):
            response["translation"] = result["translation"]
            
        return response
        
    except Exception as e:
        logger.error(f"Error in create_summarization: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при обработке файла. Пожалуйста, попробуйте позже."
        )

@router.post("/summarize/public", response_model=SummarizationRequestResponse)
async def create_public_summarization(
    file: UploadFile = File(...),
    language: str = Form(...),
    db: Session = Depends(get_db)
):
    """Create a new summarization request for unauthenticated users."""
    try:
        # Валидация файла
        validate_file(file)
        
        # Обработка файла и получение результатов
        result = await process_media_file(file, language)
        
        # Создание записи в базе данных с user_id=0
        request = SummarizationRequest(
            user_id=0,
            filename=file.filename,
            transcript=result["transcription"],
            summary=result["summary"]
        )
        db.add(request)
        db.commit()
        db.refresh(request)
        
        response = {
            "id": request.id,
            "filename": request.filename,
            "transcript": request.transcript,
            "summary": request.summary,
            "created_at": request.created_at
        }
        
        if result.get("translation"):
            response["translation"] = result["translation"]
            
        return response
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating public summarization request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your request. Please try again later."
        )

@router.get("/requests")
async def get_requests(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        current_user = await get_current_user(token, db)
        requests = db.query(SummarizationRequest).filter(
            SummarizationRequest.user_id == current_user.id
        ).order_by(SummarizationRequest.created_at.desc()).all()
        
        return [
            {
                "id": request.id,
                "filename": request.filename,
                "status": request.status,
                "transcript": request.transcript,
                "summary": request.summary,
                "created_at": request.created_at
            }
            for request in requests
        ]
    except Exception as e:
        logger.error(f"Error in get_requests: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail="Необходима авторизация для просмотра истории запросов"
        )

@router.get("/check-auth")
async def check_auth(current_user: User = Depends(get_current_user)):
    """Check if the user is authenticated and return their email and name."""
    return {
        "is_authenticated": True,
        "email": current_user.email,
        "name": current_user.name
    }

@router.post("/translate")
async def translate(
    text: str = Form(...),
    target_language: str = Form(...),
    current_user: UserResponse = Depends(get_current_user)
):
    """
    Переводит текст на указанный язык
    """
    try:
        translated_text = translate_text(text, target_language)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 