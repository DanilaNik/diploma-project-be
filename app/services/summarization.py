import os
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable
import torch
from sqlalchemy.orm import Session
from fastapi import HTTPException, UploadFile
from app.models.models import SummarizationRequest
import logging
import io
import tempfile
import numpy as np
import soundfile as sf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import ffmpeg
import subprocess
from app.core.config import settings, settings_model, LANGUAGE_PREFIXES, LANGUAGE_NAMES
import multiprocessing
import gc
from app.services.transcription import TranscriptionService
from app.services.text_processing import TextProcessor
import functools

logger = logging.getLogger(__name__)

transcription_service = None
text_processor = None

def get_transcription_service():
    global transcription_service
    if transcription_service is None:
        transcription_service = TranscriptionService(
            model_size=settings_model.WHISPER_MODEL_SIZE
        )
    return transcription_service

def get_text_processor():
    global text_processor
    if text_processor is None:
        text_processor = TextProcessor(
            cache_dir=settings_model.MODEL_CACHE_DIR,
            device=settings_model.DEVICE
        )
    return text_processor

executor = ThreadPoolExecutor(max_workers=max(2, multiprocessing.cpu_count()))

async def run_in_threadpool(func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, functools.partial(func, *args, **kwargs)
    )

def validate_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_extension}' not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    file_size = 0
    for chunk in file.file:
        file_size += len(chunk)
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size / (1024 * 1024):.2f}MB). Maximum size is {settings.MAX_FILE_SIZE / (1024 * 1024)}MB"
            )
    file.file.seek(0)

def extract_audio_from_memory(file_content: bytes) -> str:
    temp_input_path = None
    temp_output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as temp_input:
            temp_input.write(file_content)
            temp_input_path = temp_input.name

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_output:
            temp_output_path = temp_output.name

        command = [
            "ffmpeg",
            "-i", temp_input_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "32k",
            "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25,dynaudnorm=f=75:g=25",
            "-threads", "4",
            "-y",
            temp_output_path
        ]

        subprocess.run(command, check=True, capture_output=True)

        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

        return temp_output_path
    except Exception as e:
        logger.error(f"Error in extract_audio_from_memory: {str(e)}", exc_info=True)
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_output_path and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
        raise

def transcribe_audio(audio_path: str, language: str = "ru") -> str:
    try:
        service = get_transcription_service()
        segments = service.transcribe_file(
            audio_path=audio_path,
            language=language
        )
        return service.merge_segments(segments)
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}", exc_info=True)
        raise

def split_into_sentences(text: str) -> List[str]:
    exceptions = r'(?<!г)(?<!гг)(?<!руб)(?<!тыс)(?<!млн)(?<!млрд)(?<!см)(?<!им)(?<!т)(?<!д)(?<!проф)(?<!доц)(?<!акад)(?<!св)(?<!ул)(?<!пр)(?<!др)'
    pattern = f'{exceptions}[.!?]+'
    
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    min_length = 5
    result = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) < min_length and result:
            result[-1] = result[-1] + " " + sentence
        else:
            result.append(sentence)
    
    return result

def get_token_count(text: str) -> int:
    words = text.split()
    return len(words) // 4 + 1

def create_chunks_with_overlap(sentences: List[str], max_tokens: int = 800, overlap: int = 50) -> List[str]:
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = get_token_count(sentence)
        
        if current_length + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(get_token_count(s) for s in current_chunk)
            else:
                chunks.append(sentence)
                current_chunk = []
                current_length = 0
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def summarize_text(text: str, language: str) -> str:
    try:
        logger.info(f"Summarizing text in {language}, text length: {len(text)} chars")
        
        if language not in LANGUAGE_PREFIXES:
            logger.warning(f"Unsupported language: {language}, falling back to English")
            language = "en"
            
        processor = get_text_processor()
        result = processor.summarize(
            text=text,
            language=language,
            max_length=settings_model.SUMMARIZATION_MAX_LENGTH,
            min_length=settings_model.SUMMARIZATION_MIN_LENGTH,
            temperature=settings_model.SUMMARIZATION_TEMPERATURE
        )
        
        logger.info(f"Summarization completed, result length: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}", exc_info=True)
        raise

def translate_text(text: str, target_language: str, source_language: str = None) -> str:
    try:
        source_language = source_language or "en"
        
        logger.info(f"Translating text from {source_language} to {target_language}, text length: {len(text)} chars")
        
        if target_language not in LANGUAGE_NAMES:
            logger.warning(f"Unsupported target language: {target_language}, falling back to Russian")
            target_language = "ru"
            
        if source_language not in LANGUAGE_NAMES:
            logger.warning(f"Unsupported source language: {source_language}, falling back to English")
            source_language = "en"
            
        processor = get_text_processor()
        result = processor.translate(
            text=text,
            source_lang=source_language,
            target_lang=target_language
        )
        
        logger.info(f"Translation completed, result length: {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Error in translate_text: {str(e)}", exc_info=True)
        raise

def process_chunk(chunk: str, language: str) -> str:
    try:
        processor = get_text_processor()
        prefix = LANGUAGE_PREFIXES.get(language, LANGUAGE_PREFIXES['en'])
        if language == 'ru':
            prompt = f"""Сделай подробное и информативное краткое содержание следующего текста. 
            Сохрани основные идеи, ключевые моменты и важные детали.
            Текст должен быть понятным и связным.
            
            Текст: {chunk}
            
            Краткое содержание:"""
        else:
            prompt = f"""Create a detailed and informative summary of the following text.
            Preserve main ideas, key points, and important details.
            The text should be coherent and well-structured.
            
            Text: {chunk}
            
            Summary:"""
            
        return processor.summarize(
            text=chunk,
            language=language,
            max_length=settings_model.SUMMARIZATION_MAX_LENGTH,
            min_length=settings_model.SUMMARIZATION_MIN_LENGTH,
            temperature=0.3
        )
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        return chunk

def process_chunks_parallel(chunks: List[str], language: str) -> List[str]:
    max_workers = min(len(chunks), multiprocessing.cpu_count() * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, chunk, language) for chunk in chunks]
        results = [future.result() for future in futures]
    return results

def merge_summaries(summaries: List[str], language: str) -> str:
    if len(summaries) <= 1:
        return summaries[0] if summaries else ""
        
    processor = get_text_processor()
    
    if language == 'ru':
        prompt = f"""Объедини следующие краткие описания в одно подробное и связное краткое содержание.
        Сохрани все важные детали, основные мысли и ключевые моменты.
        Убери повторения и сделай текст логически структурированным.
        
        Тексты для объединения:
        {" ".join(summaries)}
        
        Итоговое краткое содержание:"""
    else:
        prompt = f"""Merge the following summaries into one detailed and coherent summary.
        Preserve all important details, main ideas, and key points.
        Remove repetitions and make the text logically structured.
        
        Texts to merge:
        {" ".join(summaries)}
        
        Final summary:"""
    
    return processor.summarize(
        text=" ".join(summaries),
        language=language,
        max_length=settings_model.SUMMARIZATION_MAX_LENGTH,
        min_length=settings_model.SUMMARIZATION_MIN_LENGTH,
        temperature=0.3
    )

def generate_summary(text: str, language: str = "ru") -> str:
    try:
        clear_gpu_memory()
        
        sentences = split_into_sentences(text)
        
        if len(sentences) < 5:
            return process_chunk(text, language)
            
        chunks = create_chunks_with_overlap(sentences)
        
        logger.info(f"Created {len(chunks)} chunks for summarization")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1} length: {len(chunk.split())} words")
        
        chunk_summaries = process_chunks_parallel(chunks, language)
        
        logger.info(f"Generated {len(chunk_summaries)} summaries")
        for i, summary in enumerate(chunk_summaries):
            logger.info(f"Summary {i+1} length: {len(summary.split())} words")
        
        final_summary = merge_summaries(chunk_summaries, language)
        
        logger.info(f"Final summary length: {len(final_summary.split())} words")
        
        clear_gpu_memory()
        
        return final_summary
        
    except Exception as e:
        logger.error(f"Error in generate_summary: {str(e)}", exc_info=True)
        raise

async def process_media_file(file: UploadFile, language: str) -> dict:
    try:
        validate_file(file)
        
        if language not in LANGUAGE_PREFIXES:
            logger.warning(f"Unsupported language: {language}, falling back to English")
            language = "en"
        
        logger.info(f"Processing media file in language: {language}")
        
        content = await file.read()
        
        audio_path = await run_in_threadpool(extract_audio_from_memory, content)
        
        try:
            logger.info(f"Starting transcription in {language}")
            transcription = await run_in_threadpool(transcribe_audio, audio_path, language)
            logger.info(f"Transcription completed, length: {len(transcription)} chars")
            
            logger.info(f"Starting summarization in {language}")
            summary = await run_in_threadpool(generate_summary, transcription, language)
            logger.info(f"Summarization completed, length: {len(summary)} chars")
            
            translation = None
            if language != "ru":
                logger.info(f"Starting translation from {language} to Russian")
                translation = await run_in_threadpool(translate_text, summary, "ru", language)
                logger.info(f"Translation completed, length: {len(translation)} chars")
            
            return {
                "transcription": transcription,
                "summary": summary,
                "translation": translation
            }
            
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            
            await run_in_threadpool(clear_gpu_memory)
            
    except Exception as e:
        logger.error(f"Error in process_media_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 