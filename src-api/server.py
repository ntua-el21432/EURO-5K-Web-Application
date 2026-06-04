from os import path, environ
from transformers import pipeline
import torch
from fastapi import FastAPI, HTTPException, Header, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import logging
import asyncio
import gc
import time
from typing import Dict
from contextlib import asynccontextmanager

import numpy as np
from lime.lime_text import LimeTextExplainer

from legalSplitter import split_sentences
from rdf_utils import generate_rrmv_turtle
from model_utils import load_bert_model, load_model_unsloth, load_model_transformers
from dotenv import load_dotenv

# Set multithreading for faster explanations
torch.set_num_threads(5)
torch.set_num_interop_threads(1)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Configuration
MODEL_IDLE_TIMEOUT = int(environ.get("MODEL_IDLE_TIMEOUT", "300"))  # 5 min default
MAX_CACHE_SIZE = int(environ.get("MAX_CACHE_SIZE", "2"))  # Keep max 2 models hot

MODEL_CONFIGURATIONS = {
    'bert_base_fft': {'model_in': 'bert-base-uncased', 'loader': 'transformers', 'task': 'token-classification', 'num_labels': 3, 'is_peft': False},
    'bert_eurlex_fft': {'model_in': 'nlpaueb/bert-base-uncased-eurlex', 'loader': 'transformers', 'task': 'token-classification', 'num_labels': 3, 'is_peft': False},
    'bert_base_lora': {'model_in': 'bert-base-uncased', 'loader': 'transformers', 'task': 'token-classification', 'num_labels': 3, 'is_peft': False},
    'bert_eurlex_lora': {'model_in': 'nlpaueb/bert-base-uncased-eurlex', 'loader': 'transformers', 'task': 'token-classification', 'num_labels': 3, 'is_peft': False},
    'mistral': {'model_in': 'unsloth/mistral-7b-v0.3-bnb-4bit', 'template_type': 'mistral', 'loader': 'unsloth', 'task': 'text-generation', 'num_labels': 2, 'is_peft': True},
    'saul': {'model_in': 'Equall/Saul-7B-Base', 'template_type': 'mistral', 'loader': 'transformers', 'requires_manual_quantization': True, 'task': 'text-generation', 'num_labels': 2, 'is_peft': True},
    'llama': {'model_in': 'unsloth/Meta-Llama-3.1-8B-bnb-4bit', 'template_type': 'llama-3.1', 'loader': 'unsloth', 'task': 'text-generation', 'num_labels': 2, 'is_peft': True}
}

STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'when', 'where', 'why', 'how', 'not', 'no',
    'nor', 'so', 'than', 'such', 'both', 'few', 'more', 'most', 'some',
    'any', 'each', 'every', 'either', 'neither', 'other', 'another'
}

POSITIVE_LABEL = "LABEL_1"
OBLIGATION_LABELS = ["B-RO", "I-RO"]
NER_MODELS = ["bert_base_fft", "bert_eurlex_fft", "bert_base_lora", "bert_eurlex_lora"]

# LIME parameters
NUM_LIME_SAMPLES = 2000
NUM_FEATURES = 50

# LIME helpers
def is_meaningful(token):
    clean = token.strip().lower()
    if not clean or len(clean) < 3:
        return False
    if clean in STOPWORDS:
        return False
    return clean.isalpha()

def generate_custom_html(text, exp_list, model_name, classification="Live Prediction", sentence_id="N/A"):
    logger.info(f"Generating HTML for text: {text[:30]}...")
    logger.info(f"Top features passed to HTML: {exp_list[:5]}")

    tokens = text.split()
    attr_dict = {token: score for token, score in exp_list}
    html_tokens = []
    for token in tokens:
        score = attr_dict.get(token, 0.0)
        if not is_meaningful(token):
            style = 'background-color: #f5f5f5; color: #666; border: 1px solid #ddd;'
        elif abs(score) < 0.01:
            style = 'background-color: #f5f5f5; color: #666; border: 1px solid #ddd;'
        else:
            if score > 0:
                intensity = min(abs(score) * 5, 1.0)
                alpha = 0.15 + intensity * 0.65
                style = f'background-color: rgba(255, 0, 0, {alpha}); color: #000; border: 2px solid #d00; font-weight: bold;'
            else:
                intensity = min(abs(score) * 5, 1.0)
                alpha = 0.15 + intensity * 0.65
                style = f'background-color: rgba(0, 0, 255, {alpha}); color: #000; border: 2px solid #00d; font-weight: bold;'

        html_tokens.append(
            f'<span style="{style} padding: 4px 7px; margin: 2px; cursor: help; '
            f'border-radius: 4px; display: inline-block;" '
            f'data-bs-toggle="tooltip" data-bs-placement="top" '
            f'title="Score: {score:.4f}">{token}</span>'
        )

    return f'''
    <div style="font-family: 'Segoe UI', Arial, sans-serif; background: #fff; padding: 20px; border-radius: 8px;">
        <h2 style="color: #222; border-bottom: 2px solid #ddd; padding-bottom: 10px;">LIME Explanation: {model_name.upper()}</h2>
        <div style="font-size: 14px; color: #555; margin: 15px 0;"><strong>Classification:</strong> {classification}</div>
        <div style="margin: 15px 0; padding: 12px; background: #fafafa; border-left: 3px solid #999; border-radius: 4px;">
            <em style="color: #555; font-size: 14px;">{text}</em>
        </div>
        <div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px;">
            <strong>Color Legend:</strong><br>
            <div style="margin-top: 10px; line-height: 1.8;">
                <span style="display: inline-block; background-color: rgba(255, 0, 0, 0.6); color: #000; border: 2px solid #d00; font-weight: bold; padding: 2px 8px; border-radius: 4px;">Red</span>
                <span style="margin-left: 5px;">= Positive (supports RO)</span> |

                <span style="display: inline-block; background-color: rgba(0, 0, 255, 0.6); color: #000; border: 2px solid #00d; font-weight: bold; padding: 2px 8px; border-radius: 4px; margin-left: 10px;">Blue</span>
                <span style="margin-left: 5px;">= Negative (against RO)</span> |

                <span style="display: inline-block; background-color: #f5f5f5; color: #666; border: 1px solid #ddd; font-weight: bold; padding: 2px 8px; border-radius: 4px; margin-left: 10px;">Gray</span>
                <span style="margin-left: 5px;">= Neutral</span>
            </div>
            <br>
            <em style="font-size: 0.9em; color: #666;">Hover to see LIME score</em>
        </div>
        <div style="font-size: 15px; line-height: 2.5; margin: 20px 0;">{' '.join(html_tokens)}</div>
    </div>
    '''

class BERTClassifierWrapper:
    """Wrapper to make BERT compatible with LIME"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = 64
        self.model.eval()

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        probs_list = []
        for sent in sentences:
            tokens = sent.split()
            inputs = self.tokenizer(tokens, is_split_into_words=True,
                                   return_tensors='pt', truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                prob_pos = probs[:, :, 0].max().item()
                probs_list.append([1 - prob_pos, prob_pos])

        return np.array(probs_list)

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._access_order: list = []

    def _get_lock(self, key: str):
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    def _evict_oldest(self, current_key: str):
        """Evict oldest model if at capacity, excluding current key"""
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Find oldest (first in access_order that's not current)
            oldest_key = None
            for k in self._access_order:
                if k != current_key and k in self._cache:
                    oldest_key = k
                    break

            if oldest_key:
                logger.info(f"Cache full ({len(self._cache)}/{MAX_CACHE_SIZE}), evicting {oldest_key}")
                self.unload(oldest_key)
                return True
        return False

    async def get(self, key: str):
        """Get model from cache or load it"""
        # Cache hit - update LRU and return immediately
        if key in self._cache:
            self._cache[key]["last_used"] = time.time()
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Cache hit for {key} (current size: {len(self._cache)})")
            return self._cache[key]["classifier"], self._cache[key]["raw_model"]

        # Cache miss - acquire lock and load
        async with self._get_lock(key):
            # Double-check after acquiring lock
            if key in self._cache:
                logger.debug(f"Cache hit for {key} (after lock)")
                return self._cache[key]["classifier"], self._cache[key]["raw_model"]

            # Evict BEFORE loading to keep size in check
            if len(self._cache) >= MAX_CACHE_SIZE:
                self._evict_oldest(key)

            # Now load
            await self._load(key)

        return self._cache[key]["classifier"], self._cache[key]["raw_model"]

    async def _load(self, key: str):
        """Blocking model loading run in thread pool"""
        if key not in MODEL_CONFIGURATIONS:
            raise HTTPException(status_code=400, detail=f"Model '{key}' not configured")

        if key not in lang_mappings:
            raise HTTPException(status_code=400, detail=f"Model '{key}' not in mappings")

        config = MODEL_CONFIGURATIONS[key]
        mapping_entry = lang_mappings[key]
        model_name = mapping_entry["folder_name"] if isinstance(mapping_entry, dict) else mapping_entry
        model_dir = path.join(models_path, model_name)

        has_config = path.exists(path.join(model_dir, "config.json"))
        has_adapter = path.exists(path.join(model_dir, "adapter_config.json"))

        if not has_config and not has_adapter:
            raise HTTPException(status_code=400, detail=f"Model '{key}' files not found at {model_dir}")

        logger.info(f"Lazy-loading model {key}...")
        loop = asyncio.get_event_loop()

        def _blocking_load():
            if config['task'] == 'token-classification':
                model, tokenizer = load_bert_model(
                    model_dir,
                    num_labels=config['num_labels'],
                    id2label={0: "B-RO", 1: "I-RO", 2: "O"},
                    label2id={"B-RO": 0, "I-RO": 1, "O": 2}
                )
                classifier = pipeline(
                    task=config['task'],
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    aggregation_strategy="none",
                )
                raw = {"model": model, "tokenizer": tokenizer, "device": device}
                return classifier, raw
            else:
                try:
                    loader_type = config.get('loader', 'unsloth')
                    if loader_type == 'unsloth':
                        model, tokenizer = load_model_unsloth(model_dir)
                    else:
                        model, tokenizer = load_model_transformers(model_dir)

                    target_device = llm_device if torch.cuda.is_available() else device
                    classifier = pipeline(
                        task=config['task'],
                        model=model,
                        tokenizer=tokenizer,
                        device=target_device,
                        top_k=None,
                    )
                    raw = {"model": model, "tokenizer": tokenizer, "device": target_device, "type": "llm"}
                    return classifier, raw
                except Exception as e:
                    logger.warning(f"Failed to load LLM {key} on this device.")
                    raise HTTPException(status_code=503, detail=f"Different infrastructure required")


        classifier, raw_model = await loop.run_in_executor(None, _blocking_load)

        self._cache[key] = {
            "classifier": classifier,
            "raw_model": raw_model,
            "last_used": time.time(),
            "config": config
        }

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        logger.info(f"Model {key} loaded. Cache size: {len(self._cache)}/{MAX_CACHE_SIZE}")

    # def unload(self, key: str):
    #     """Explicitly unload a model to free memory"""
    #     if key in self._cache:
    #         logger.info(f"Unloading model {key}")
    #         del self._cache[key]
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #         if key in self._access_order:
    #             self._access_order.remove(key)
    def unload(self, key: str):
        if key in self._cache:
            logger.info(f"Unloading model {key}")

            model_data = self._cache[key]
                # Explicit deletes
            del model_data["classifier"]
            del model_data["raw_model"]
            del self._cache[key]
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extra (important)
            import ctypes
            libc  = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)

    async def cleanup_idle(self):
        """Background task: evict idle models"""
        while True:
            await asyncio.sleep(60)
            now = time.time()
            to_unload = []

            for key, data in list(self._cache.items()):
                if now - data["last_used"] > MODEL_IDLE_TIMEOUT:
                    to_unload.append(key)

            # Also enforce hard limit if somehow exceeded
            while len(self._cache) > MAX_CACHE_SIZE:
                # Find oldest not already marked for unloading
                for key in self._access_order:
                    if key not in to_unload and key in self._cache:
                        to_unload.append(key)
                        break
                else:
                    break

            for key in to_unload:
                self.unload(key)

            if to_unload:
                logger.info(f"Cleanup complete. Active models: {list(self._cache.keys())}")

# Global cache instance
model_cache = ModelCache()

# Environment setup
models_path = environ.get("MODEL_PATH", "./models")
device = environ.get("DEVICE", "cpu")
llm_device = environ.get("LLM_DEVICE", "cuda")
ui_path = environ.get("UI_PATH", "./dist")
lang_mappings_path = environ.get("MAPPINGS_PATH", "./mappings.json")

lang_mappings = {}
if path.exists(lang_mappings_path):
    with open(lang_mappings_path, "r") as f:
        lang_mappings = json.load(f)
        logger.info(f"Loaded {len(lang_mappings)} model mappings")
else:
    logger.warning(f"Mappings file not found: {lang_mappings_path}")


AVAILABLE_MODEL_KEYS = set()
for k in MODEL_CONFIGURATIONS:
    if k not in lang_mappings:
        continue
    mapping_entry = lang_mappings[k]
    model_name = mapping_entry["folder_name"] if isinstance(mapping_entry, dict) else mapping_entry
    model_dir = path.join(models_path, model_name)

    if path.exists(path.join(model_dir, "config.json")) or \
       path.exists(path.join(model_dir, "adapter_config.json")):
        AVAILABLE_MODEL_KEYS.add(k)

logger.info(f"Available models: {AVAILABLE_MODEL_KEYS}")

logger.info("API configuration complete. Models will be loaded on demand.")

# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start cleanup task
    cleanup_task = asyncio.create_task(model_cache.cleanup_idle())
    yield
    # Shutdown: cleanup
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    # Unload all models
    for key in list(model_cache._cache.keys()):
        model_cache.unload(key)

middleware = [Middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])]
app = FastAPI(middleware=middleware, lifespan=lifespan)

class TextRequest(BaseModel):
    text: str = ""
    model: str
    threshold: float = 0.5
    shouldSplitSent: bool = True

class ExplainRequest(BaseModel):
    text: str = ""
    model: str

class ExportRequest(BaseModel):
    predictions: list

@app.get("/api/models")
async def get_models():
    """Return all models - active status from startup cache"""
    return {
        k: {
            "name": k,
            "active": k in AVAILABLE_MODEL_KEYS,
            "type": MODEL_CONFIGURATIONS[k].get('task'),
            "loaded": k in model_cache._cache
        }
        for k in MODEL_CONFIGURATIONS  # Iterate all configs
    }

@app.post("/api/predict")
async def post_data(request: TextRequest, Token: str = Header(None, convert_underscores=False)):
    if environ.get("AUTH_TOKEN") and Token != environ.get("AUTH_TOKEN"):
        raise HTTPException(status_code=500, detail="Wrong token header")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # Lazy load model
    try:
        classifier, raw_model = await model_cache.get(request.model)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' does not exist or failed to load")

    sentences = split_sentences(request.text) if request.shouldSplitSent else [request.text]
    if not sentences:
        return {"predictions": [], "obligation_count": 0}

    good_predictions = []
    model_key = request.model

    if model_key in NER_MODELS:
        model = raw_model["model"]
        tokenizer = raw_model["tokenizer"]
        device_model = raw_model["device"]

        for i, sentence in enumerate(sentences):
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device_model)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                token_probs = probs[0]
                b_ro_score = token_probs[:, model.config.label2id["B-RO"]].max().item()
            is_obligation = b_ro_score > request.threshold
            good_predictions.append({
                "sentence_index": i,
                "text": sentence,
                "is_reporting_obligation": is_obligation,
                "score": float(b_ro_score)
            })
    else:
        all_predictions = classifier(sentences)

        for i, preds in enumerate(all_predictions):
            is_obligation = False
            score = 0.0
            loop_preds = preds if isinstance(preds, list) else [preds]
            for p in loop_preds:
                if p['label'] == POSITIVE_LABEL:
                    score = p['score']
                    if score > request.threshold:
                        is_obligation = True
                    break

            good_predictions.append({
                "sentence_index": i,
                "text": sentences[i],
                "is_reporting_obligation": is_obligation,
                "score": score
            })

    obligation_count = sum(1 for p in good_predictions if p['is_reporting_obligation'])
    return {"predictions": good_predictions, "obligation_count": obligation_count}

@app.post("/api/explain")
async def explain_prediction(request: ExplainRequest, Token: str = Header(None, convert_underscores=False)):
    if environ.get("AUTH_TOKEN") and Token != environ.get("AUTH_TOKEN"):
        raise HTTPException(status_code=500, detail="Wrong token header")

    # Lazy load model
    try:
        classifier, raw_model = await model_cache.get(request.model)
    except HTTPException:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' does not exist or failed to load")

    text = request.text.strip()
    model_key = request.model
    config = MODEL_CONFIGURATIONS.get(model_key)
    task_type = config.get('task', 'text-classification')

    if model_key in NER_MODELS:
        wrapper = BERTClassifierWrapper(
            model=raw_model["model"],
            tokenizer=raw_model["tokenizer"],
            device=raw_model["device"]
        )
        predictor = wrapper
    else:
        def predictor(texts):
            if hasattr(texts, 'tolist'):
                texts = texts.tolist()
            texts = list(texts)
            res = []
            try:
                preds = classifier(texts, batch_size=32, top_k=None)
                for p_list in preds:
                    loop_p = p_list if isinstance(p_list, list) else [p_list]
                    score = 0.0
                    for p in loop_p:
                        if p['label'] == POSITIVE_LABEL:
                            score = p['score']
                            break
                    res.append([1-score, score])
            except Exception as e:
                logger.error(f"Prediction error in explainer: {e}")
                res = [[1.0, 0.0] for _ in texts]
            return np.array(res)

    explainer = LimeTextExplainer(class_names=["Not Obligation", "Obligation"])
    exp = explainer.explain_instance(text, predictor, num_features=NUM_FEATURES, num_samples=NUM_LIME_SAMPLES)
    logger.info(f"LIME DEBUG: Top attributes: {exp.as_list()}")

    return {"html_content": generate_custom_html(text, exp.as_list(), model_key)}

@app.post("/api/export/rdf")
async def export_rdf(request: ExportRequest, Token: str = Header(None, convert_underscores=False)):
    if environ.get("AUTH_TOKEN") and Token != environ.get("AUTH_TOKEN"):
        raise HTTPException(status_code=500, detail="Wrong token header")

    obligations = [p for p in request.predictions if p.get('is_reporting_obligation')]
    if not obligations:
        raise HTTPException(status_code=400, detail="No obligations found to export.")

    try:
        turtle_data = generate_rrmv_turtle(obligations)
        return Response(
            content=turtle_data,
            media_type="text/turtle",
            headers={
                "Content-Disposition": "attachment; filename=obligations_rrmv.ttl"
            }
        )
    except Exception as e:
        logger.error(f"RDF Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if path.exists(ui_path):
    app.mount("/ui", StaticFiles(directory=ui_path, html=True), name="ui")


@app.get("/health")
async def health():
    return {"status": "ok"}