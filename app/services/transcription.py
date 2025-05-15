import os
import json
import subprocess
from typing import List, Optional
from pathlib import Path
import ffmpeg
from app.core.config import settings_model
import logging
import time
import psutil
from tempfile import NamedTemporaryFile
import re

logger = logging.getLogger(__name__)

def log_system_stats():
    """Логирование системных ресурсов"""
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_percent}% | RAM: {memory.percent}% | Available RAM: {memory.available / 1024 / 1024:.0f}MB")

class TranscriptionService:
    def __init__(
        self,
        model_size: Optional[str] = None,
        compute_type: str = "float16"
    ):
        logger.info("🚀 Initializing whisper model...")
        log_system_stats()
        
        # Используем мультиязычную модель среднего размера
        self.model_path = os.path.join(os.path.dirname(__file__), "../models/ggml-medium.bin")
        self.whisper_cli = os.path.join(os.path.dirname(__file__), "../lib/whisper/build/bin/whisper-cli")
        
        # Проверяем наличие бинарника
        if not os.path.exists(self.whisper_cli):
            logger.info("Building whisper.cpp...")
            self._build_whisper_cpp()
        
        # Проверяем наличие модели
        if not os.path.exists(self.model_path):
            logger.info("Downloading whisper model...")
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            download_script = os.path.join(os.path.dirname(__file__), "../lib/whisper/models/download-ggml-model.sh")
            subprocess.run(["sh", download_script, "medium", model_dir], check=True)
        
        logger.info("✅ Whisper model initialized successfully")
        log_system_stats()

    def _build_whisper_cpp(self):
        """Собирает whisper.cpp с поддержкой Metal"""
        whisper_dir = os.path.join(os.path.dirname(__file__), "../lib/whisper")
        
        # Создаем build директорию
        os.makedirs(os.path.join(whisper_dir, "build"), exist_ok=True)
        
        # Собираем проект с поддержкой Metal
        subprocess.run(["cmake", "-B", "build", "-DWHISPER_METAL=ON"], cwd=whisper_dir, check=True)
        subprocess.run(["cmake", "--build", "build", "--config", "Release"], cwd=whisper_dir, check=True)

    def transcribe_file(
        self, 
        audio_path: str,
        language: Optional[str] = None,
        beam_size: Optional[int] = None
    ) -> List[dict]:
        """Транскрибирует файл используя whisper.cpp с Metal ускорением."""
        try:
            logger.info(f"\n{'='*50}\n🎯 Starting transcription: {audio_path}\n{'='*50}")
            log_system_stats()
            
            beam_size = beam_size or settings_model.TRANSCRIPTION_BEAM_SIZE
            
            # Создаем временный файл для JSON вывода
            with NamedTemporaryFile(suffix='.json') as tmp_json:
                # Базовые параметры для whisper.cpp
                cmd = [
                    self.whisper_cli,
                    "-m", self.model_path,
                    "-f", audio_path,
                    "-oj",  # JSON output
                    "-of", tmp_json.name[:-5],  # Убираем .json расширение
                    "-pp",  # Показывать прогресс
                    "-pc",  # Цветной вывод
                    "--print-progress",  # Подробный вывод прогресса
                    "--print-special",   # Вывод специальных токенов
                    "-t", "4",           # Количество потоков
                    "-bs", str(beam_size),  # Размер луча для поиска
                    "-bo", "5",          # Количество лучших вариантов
                    "-wt", "0.01",       # Порог определения слов
                    "-ml", "0",          # Без ограничения длины сегмента
                    "-ac", "0",          # Использовать весь аудио контекст
                    "--temperature", "0.0"  # Температура сэмплирования
                ]
                
                if language:
                    cmd.extend(["-l", language])
                    # Для русского языка добавляем prompt на русском
                    if language.lower() == 'ru':
                        cmd.extend(["--prompt", "Далее идет русская речь"])
                
                logger.info(f"Running whisper.cpp command: {' '.join(cmd)}")
                logger.info(f"System info before transcription:")
                log_system_stats()
                
                start_time = time.time()
                
                # Запускаем процесс
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                # Логируем вывод
                if process.stdout:
                    logger.info(f"Whisper output: {process.stdout}")
                if process.stderr:
                    logger.warning(f"Whisper stderr: {process.stderr}")
                
                if process.returncode != 0:
                    logger.error(f"Whisper CLI error: {process.stderr}")
                    raise Exception(f"Transcription failed: {process.stderr}")
                
                # Проверяем существование файла
                if not os.path.exists(tmp_json.name):
                    logger.error(f"JSON output file not found: {tmp_json.name}")
                    raise Exception("Transcription output file not found")
                
                # Читаем результаты из JSON файла
                try:
                    with open(tmp_json.name, 'r') as f:
                        results = json.load(f)
                except json.JSONDecodeError as e:
                    # Читаем содержимое файла для отладки
                    with open(tmp_json.name, 'r') as f:
                        content = f.read()
                    logger.error(f"JSON file content: {content}")
                    logger.error(f"Failed to parse JSON output: {str(e)}")
                    raise Exception("Failed to parse transcription output")
                
                # Логируем результаты
                logger.info(f"Transcription results: {results}")
                
                # Конвертируем в нужный формат и проверяем наличие текста
                segments = []
                for segment in results.get('transcription', []):
                    # Получаем текст из сегмента и нормализуем пробелы
                    text = segment.get('text', '').strip()
                    
                    # Очищаем текст от всех специальных токенов
                    text = text.replace('[_BEG_]', '').strip()
                    # Удаляем токены вида [_TT_XXX]
                    text = re.sub(r'\[_TT_\d+\]', '', text).strip()
                    # Нормализуем множественные пробелы
                    text = ' '.join(text.split())
                    
                    # Пропускаем пустые сегменты
                    if text:
                        try:
                            # Конвертируем временные метки в секунды
                            def time_to_seconds(time_str: str) -> float:
                                try:
                                    # Формат входной строки: "00:00:00,000"
                                    main_part, ms_part = time_str.split(',')
                                    h, m, s = map(int, main_part.split(':'))
                                    return h * 3600 + m * 60 + s + int(ms_part) / 1000
                                except (ValueError, IndexError) as e:
                                    logger.warning(f"Failed to parse timestamp {time_str}: {e}")
                                    return 0.0

                            start_time = time_to_seconds(segment['timestamps']['from'])
                            end_time = time_to_seconds(segment['timestamps']['to'])

                            segments.append({
                                'start': start_time,
                                'end': end_time,
                                'text': text
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process segment {segment}: {e}")
                            continue
                
                # Проверяем, что есть хоть какой-то текст
                if not segments:
                    logger.warning("No valid text segments found in transcription output")
                    raise Exception("No valid text segments found in transcription output")
            
            logger.info(f"\n{'='*50}\n✅ Transcription completed\n{'='*50}")
            logger.info(f"Transcription took {time.time() - start_time:.2f} seconds")
            logger.info(f"System info after transcription:")
            log_system_stats()
            return segments
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {str(e)}")
            raise

    @staticmethod
    def merge_segments(segments: List[dict]) -> str:
        """Объединяет сегменты в единый текст."""
        return " ".join(segment['text'].strip() for segment in segments) 