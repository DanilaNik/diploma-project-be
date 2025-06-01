from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Form, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.core.config import settings
from app.db.database import get_db
from app.models.models import User, SummarizationRequest
from app.schemas.schemas import UserCreate, UserResponse, Token, SummarizationRequestResponse, RequestsSearchParams
from app.services.summarization import process_media_file, validate_file, translate_text
from app.core.security import get_password_hash, verify_password, get_current_user
from app.crud import get_user_by_email, create_user, authenticate_user
from app.auth import create_access_token

logger = logging.getLogger(__name__)
router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Пользователь с таким email уже зарегистрирован. Пожалуйста, используйте другой email или выполните вход."
            )
        
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Произошла ошибка при регистрации. Пожалуйста, попробуйте позже."
        )

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
        user_id = None
        if token:
            try:
                current_user = await get_current_user(token, db)
                user_id = current_user.id
            except Exception as e:
                logger.warning(f"Invalid token: {str(e)}")
        
        result = await process_media_file(file, language)
        
        db_request = SummarizationRequest(
            user_id=user_id,
            filename=file.filename,
            transcript=result["transcription"],
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
    try:
        validate_file(file)
        
        result = await process_media_file(file, language)
        
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

@router.post("/requests")
async def get_requests(
    search_params: RequestsSearchParams,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        current_user = await get_current_user(token, db)
        
        requests_query = db.query(SummarizationRequest).filter(
            SummarizationRequest.user_id == current_user.id
        )
        
        if search_params.filename_query:
            search_term = f"%{search_params.filename_query}%"
            requests_query = requests_query.filter(
                SummarizationRequest.filename.ilike(search_term)
            )
        
        if search_params.content_query:
            search_term = f"%{search_params.content_query}%"
            requests_query = requests_query.filter(
                SummarizationRequest.transcript.ilike(search_term)
            )
        
        requests = requests_query.order_by(SummarizationRequest.created_at.desc()).all()
        
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

@router.get("/requests/{request_id}")
async def get_request_by_id(
    request_id: int,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        current_user = await get_current_user(token, db)
        
        request = db.query(SummarizationRequest).filter(
            SummarizationRequest.id == request_id,
            SummarizationRequest.user_id == current_user.id
        ).first()
        
        if not request:
            raise HTTPException(
                status_code=404,
                detail="Запрос не найден или у вас нет доступа к нему"
            )
        
        return {
            "id": request.id,
            "filename": request.filename,
            "status": request.status,
            "transcript": request.transcript,
            "summary": request.summary,
            "created_at": request.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_request_by_id: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Ошибка при получении запроса"
        )

@router.put("/requests/{request_id}/update")
async def update_request(
    request_id: int,
    transcript: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    try:
        current_user = await get_current_user(token, db)
        
        request = db.query(SummarizationRequest).filter(
            SummarizationRequest.id == request_id,
            SummarizationRequest.user_id == current_user.id
        ).first()
        
        if not request:
            raise HTTPException(
                status_code=404,
                detail="Запрос не найден или у вас нет доступа к нему"
            )
        
        if transcript is not None:
            request.transcript = transcript
        
        if summary is not None:
            request.summary = summary
        
        db.commit()
        db.refresh(request)
        
        return {
            "id": request.id,
            "filename": request.filename,
            "status": request.status,
            "transcript": request.transcript,
            "summary": request.summary,
            "created_at": request.created_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Ошибка при обновлении запроса"
        )

@router.get("/check-auth")
async def check_auth(current_user: User = Depends(get_current_user)):
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
    try:
        translated_text = translate_text(text, target_language)
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 