from pydantic_settings import BaseSettings
from typing import List, Dict
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

load_dotenv()

LANGUAGE_PREFIXES = {
    'ru': 'Сделай краткое содержание: ',
    'en': 'Summarize: ',
    'de': 'Zusammenfassen: ',
    'fr': 'Résumer: ',
    'es': 'Resumir: ',
    'zh': '总结: '
}

LANGUAGE_NAMES = {
    'en': 'English',
    'zh': 'Chinese',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian'
}

class Settings(BaseSettings):
    PROJECT_NAME: str = "Video Summarization API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    DATABASE_URL: str
    
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 600
    
    HF_TOKEN: str
    
    MODEL_CACHE_DIR: str = "/app/model_cache"
    
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    SSL_KEYFILE: str = "key.pem"
    SSL_CERTFILE: str = "cert.pem"
    
    COOKIE_SECURE: bool = True
    COOKIE_HTTPONLY: bool = True
    COOKIE_SAMESITE: str = "Strict"
    
    MAX_FILE_SIZE: int = 200 * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".mp4", ".mp3", ".wav", ".avi"}

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

class ModelSettings(BaseSettings):
    MODEL_CACHE_DIR: str = "./model_cache"
    
    WHISPER_MODEL_SIZE: str = "small"
    RUSSIAN_MODEL: str = "ai-forever/FRED-T5-base"
    MT5_MODEL: str = "google/mt5-base"
    ENABLE_8BIT_QUANTIZATION: bool = True
    
    DEVICE: str = "mps"
    
    TRANSCRIPTION_CHUNK_LENGTH: int = 300
    TRANSCRIPTION_BEAM_SIZE: int = 1
    
    SUMMARIZATION_MAX_LENGTH: int = 150
    SUMMARIZATION_MIN_LENGTH: int = 50
    SUMMARIZATION_TEMPERATURE: float = 0.3
    
    class Config:
        env_prefix = "MODEL_"
        env_file = ".env"
        case_sensitive = True
        extra = "allow" 

settings = Settings()
settings_model = ModelSettings() 