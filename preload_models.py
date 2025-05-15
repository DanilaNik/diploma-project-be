import os
import torch
import gc
import psutil
from app.services.transcription import TranscriptionService
from app.services.text_processing import TextProcessor
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def clear_memory():
    """Clear memory and cache"""
    logger.info(f"Memory usage before clearing: {get_memory_usage():.2f} GB")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Memory usage after clearing: {get_memory_usage():.2f} GB")

def main():
    try:
        # Создаем директорию для кэша моделей
        cache_dir = Path("./model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        logger.info("Определение доступного устройства...")
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {device}")
        
        logger.info(f"Начальное использование памяти: {get_memory_usage():.2f} GB")
        
        # Загрузка Faster Whisper
        logger.info("\nЗагрузка Faster Whisper...")
        transcriber = TranscriptionService(
            model_size="large-v3",
            device=device
        )
        logger.info("Faster Whisper загружен успешно!")
        clear_memory()
        
        # Загрузка моделей для обработки текста
        logger.info("\nЗагрузка моделей для обработки текста...")
        text_processor = TextProcessor(
            cache_dir=cache_dir,
            device=device,
            load_in_8bit=True if device != "cpu" else False
        )
        logger.info("Модели для обработки текста загружены успешно!")
        clear_memory()
        
        logger.info("\nВсе модели загружены и готовы к использованию!")
        logger.info(f"Финальное использование памяти: {get_memory_usage():.2f} GB")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {str(e)}")
        raise

if __name__ == "__main__":
    main() 