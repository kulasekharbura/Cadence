import os
from typing import List, Dict
from google import genai
from google.genai import types
from app.services.llm.provider import LLMProvider
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiLLMProvider(LLMProvider):
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not configured.")
            self.client = None
        else:
            self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
            
        self.model = settings.GEMINI_LLM_MODEL

    async def generate_answer(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        if not self.client:
            raise ValueError("Gemini API key is missing. Cannot generate answers.")
            
        # Convert our {"role": "...", "content": "..."} messages to genai Content objects
        # We also pass the system_prompt in the model config.
        contents = []
        for msg in messages:
            # Gemini roles are typically 'user' and 'model'
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
            
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        )
        
        fallback_models_str = getattr(settings, "GEMINI_FALLBACK_MODELS", "")
        fallbacks = [m.strip() for m in fallback_models_str.split(",") if m.strip()]
        models_to_try = [self.model] + fallbacks

        last_error = None
        for i, current_model in enumerate(models_to_try):
            try:
                response = await self.client.aio.models.generate_content(
                    model=current_model,
                    contents=contents,
                    config=config
                )
                if i > 0:
                    logger.info(f"Fallback successful. Final model used: {current_model}")
                else:
                    logger.info(f"Generation successful. Selected model: {current_model}")
                return response.text
            except Exception as e:
                last_error = e
                status_code = getattr(e, "code", None)
                
                # Retryable errors: 429, 503, 404
                if status_code in (429, 503, 404):
                    logger.warning(f"Fallback attempted. Model {current_model} failed with {status_code}: {e}. Fallback reason/status: {status_code}")
                    continue
                else:
                    logger.error(f"Non-retryable error {status_code} with model {current_model}: {e}")
                    raise RuntimeError(f"Gemini LLM API error: {e}")
                    
        logger.error(f"All configured models failed. Last error: {last_error}")
        raise RuntimeError(f"Gemini LLM API error (all models failed): {last_error}")
