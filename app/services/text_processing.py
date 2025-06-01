import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Optional
import os
import logging
import re
from app.core.config import settings_model, LANGUAGE_PREFIXES

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        try:
            self.cache_dir = cache_dir or settings_model.MODEL_CACHE_DIR
            self.device = device if device != "auto" else settings_model.DEVICE
            
            # Расширенная проверка устройства
            if self.device == "auto" or self.device == "mps":
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    # Проверяем поддержку операций
                    try:
                        # Тестовый тензор
                        test_tensor = torch.zeros(1, device="mps")
                        logger.info("MPS device is fully supported and working")
                        self.device = "mps"
                    except Exception as e:
                        logger.warning(f"MPS device found but test failed: {e}")
                        self.device = "cpu"
                else:
                    if torch.cuda.is_available():
                        self.device = "cuda"
                    else:
                        self.device = "cpu"
            
            logger.info(f"TextProcessor initialized with device: {self.device}")
            logger.info(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
            logger.info(f"PyTorch MPS built: {torch.backends.mps.is_built()}")
            
            # Создаем директорию для кэша, если её нет
            os.makedirs(self.cache_dir, exist_ok=True)
                
            # Загружаем модели
            self._load_models()
            
        except Exception as e:
            logger.error(f"Error initializing TextProcessor: {str(e)}", exc_info=True)
            raise
        
    def _load_models(self):
        try:
            logger.info(f"Loading FRED-T5 model on device: {self.device}")
            # Загрузка FRED-T5 для русского языка
            self.ru_model = AutoModelForSeq2SeqLM.from_pretrained(
                settings_model.RUSSIAN_MODEL,
                torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
                cache_dir=self.cache_dir,
                load_in_8bit=settings_model.ENABLE_8BIT_QUANTIZATION and self.device != "mps"  # 8-bit только для CUDA/CPU
            ).to(self.device)
            
            self.ru_tokenizer = AutoTokenizer.from_pretrained(
                settings_model.RUSSIAN_MODEL,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loading mT5 model on device: {self.device}")
            # Загрузка mT5 для остальных языков
            self.mt5_model = AutoModelForSeq2SeqLM.from_pretrained(
                settings_model.MT5_MODEL,
                torch_dtype=torch.float16 if self.device in ["mps", "cuda"] else torch.float32,
                cache_dir=self.cache_dir,
                load_in_8bit=settings_model.ENABLE_8BIT_QUANTIZATION and self.device != "mps"  # 8-bit только для CUDA/CPU
            ).to(self.device)
            
            self.mt5_tokenizer = AutoTokenizer.from_pretrained(
                settings_model.MT5_MODEL,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Models loaded successfully on {self.device}")
            if self.device == "mps":
                logger.info("Using float16 precision for MPS device")
            elif settings_model.ENABLE_8BIT_QUANTIZATION:
                logger.info("Using 8-bit quantization for CUDA/CPU device")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}", exc_info=True)
            raise
            
    def _clear_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
    def _chunk_text(self, text: str, max_length: int = 512) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
            
        return chunks
        
    def _clean_mt5_output(self, text: str) -> str:
        cleaned = re.sub(r'<extra_id_\d+>', '', text)
        cleaned = re.sub(r'<pad>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def summarize(
        self,
        text: str,
        language: str = "ru",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        try:
            logger.info(f"Starting summarization on device: {self.device}")
            max_length = max_length or settings_model.SUMMARIZATION_MAX_LENGTH
            min_length = min_length or settings_model.SUMMARIZATION_MIN_LENGTH
            temperature = temperature or settings_model.SUMMARIZATION_TEMPERATURE
            
            chunks = self._chunk_text(text)
            summaries = []
            
            if language == "ru":
                model = self.ru_model
                tokenizer = self.ru_tokenizer
                prefix = LANGUAGE_PREFIXES.get('ru', 'Сделай краткое содержание: ')
            else:
                model = self.mt5_model
                tokenizer = self.mt5_tokenizer
                prefix = LANGUAGE_PREFIXES.get(language, 'Summarize: ')
                
            logger.info(f"Processing {len(chunks)} chunks with {model.__class__.__name__} for language: {language}, using prefix: {prefix}")
            
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                    start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
                    end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
                    
                    if start_time:
                        start_time.record()
                    
                    input_text = prefix + chunk
                    logger.info(f"Input text prefix: '{prefix[:20]}...'")
                    
                    inputs = tokenizer(
                        input_text,
                        max_length=1024,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        generation_params = {
                            "max_length": max_length,
                            "min_length": min_length,
                            "do_sample": True,
                            "temperature": temperature,
                            "num_beams": 4,
                            "no_repeat_ngram_size": 2,
                            "top_k": 50,
                            "top_p": 0.9
                        }
                        
                        if language != "ru":
                            generation_params["num_beams"] = 5
                            generation_params["length_penalty"] = 1.0
                            generation_params["early_stopping"] = True
                        
                        outputs = model.generate(
                            **inputs,
                            **generation_params
                        )
                    
                    if end_time:
                        end_time.record()
                        torch.cuda.synchronize()
                        logger.info(f"Chunk {i+1} processing time: {start_time.elapsed_time(end_time):.2f}ms")
                        
                    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if language != "ru":
                        summary = self._clean_mt5_output(summary)
                        
                    logger.info(f"Generated summary for chunk {i+1}: '{summary[:50]}...'")
                    summaries.append(summary)
                    
                    if (i + 1) % 3 == 0:
                        self._clear_gpu_memory()
                        logger.info(f"Memory cleared after chunk {i+1}")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}", exc_info=True)
                    continue
            
            logger.info("Summarization completed successfully")
            return " ".join(summaries)
            
        except Exception as e:
            logger.error(f"Error in summarize: {str(e)}", exc_info=True)
            raise
            
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "ru",
        max_length: int = 512
    ) -> str:
        try:
            logger.info(f"Starting translation from {source_lang} to {target_lang}")
            chunks = self._chunk_text(text)
            translations = []
            
            for i, chunk in enumerate(chunks):
                try:
                    input_text = f"translate {source_lang} to {target_lang}: {chunk}"
                    logger.info(f"Translation input for chunk {i+1}: '{input_text[:50]}...'")
                    
                    inputs = self.mt5_tokenizer(
                        input_text,
                        max_length=1024,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.mt5_model.generate(
                            **inputs,
                            max_length=max_length,
                            num_beams=4,
                            length_penalty=1.0,
                            no_repeat_ngram_size=2,
                            early_stopping=True
                        )
                        
                    translation = self.mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    translation = self._clean_mt5_output(translation)
                    logger.info(f"Translation output for chunk {i+1}: '{translation[:50]}...'")
                    
                    translations.append(translation)
                    
                    if (i + 1) % 3 == 0:
                        self._clear_gpu_memory()
                        logger.info(f"Memory cleared after translation chunk {i+1}")
                        
                except Exception as e:
                    logger.error(f"Error processing translation chunk {i}: {str(e)}", exc_info=True)
                    continue
                    
            result = " ".join(translations)
            logger.info(f"Translation completed, result length: {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error in translate: {str(e)}", exc_info=True)
            raise 