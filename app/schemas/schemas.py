from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class SummarizationRequestBase(BaseModel):
    original_filename: str
    file_path: str
    summary: Optional[str] = None
    status: str

class SummarizationRequestCreate(SummarizationRequestBase):
    pass

class SummarizationRequestResponse(BaseModel):
    id: int
    user_id: int
    filename: str
    transcript: str
    summary: str
    created_at: datetime

    class Config:
        from_attributes = True

class LocalFileRequest(BaseModel):
    """Schema for local file summarization request."""
    video_path: str 