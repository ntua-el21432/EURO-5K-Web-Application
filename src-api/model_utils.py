# model_utils.py
import os
import json
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForTokenClassification,
    BitsAndBytesConfig
)
from peft import PeftModel

# Try importing unsloth safely
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

logger = logging.getLogger("server")

def fix_pad_token(tokenizer, model=None, context=""):
    """Central pad token fix - handles all edge cases."""
    ctx = f" [{context}]" if context else ""
    logger.info(f" [PAD TOKEN CHECK{ctx}]")
    
    needs_fix = (
        tokenizer.pad_token is None or 
        tokenizer.pad_token == tokenizer.unk_token or
        tokenizer.pad_token_id == tokenizer.eos_token_id
    )
    
    if needs_fix:
        logger.warning(f"    ⚠ Fix needed: Setting pad_token to '<|pad|>'")
        tokenizer.pad_token = "<|pad|>"
    else:
        logger.info(f"    ✓ Pad token OK")
    
    if model is not None:
        current_model_vocab = model.get_input_embeddings().weight.size(0)
        tokenizer_vocab = len(tokenizer)
        if current_model_vocab != tokenizer_vocab:
            logger.warning(f"    ⚠ Mismatch detected: Model={current_model_vocab}, Tokenizer={tokenizer_vocab}")
            model.resize_token_embeddings(tokenizer_vocab)
            logger.info(f"    ✓ Resized embeddings")
    
    return tokenizer

def load_model_unsloth(source_path, max_seq_length=2048, load_in_4bit=True):
    """Load model using Unsloth (Native support for Llama/Mistral)."""
    if not HAS_UNSLOTH:
        logger.warning("Unsloth not found, falling back to Transformers...")
        return load_model_transformers(source_path, max_seq_length, load_in_4bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit
    )
    tokenizer = fix_pad_token(tokenizer, model, context="load_model_unsloth")
    return model, tokenizer

def load_model_transformers(source_path, max_seq_length=2048, load_in_4bit=True):
    """Load model using Transformers with manual quantization handling."""
    logger.info(f"  Loading tokenizer from {source_path}...")
    tokenizer = AutoTokenizer.from_pretrained(source_path, trust_remote_code=True)
    
    adapter_config_path = os.path.join(source_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        logger.info(f"  Detected PEFT adapter - reading config...")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_id = adapter_config.get('base_model_name_or_path')
        if not base_model_id:
            raise ValueError(f"adapter_config in {source_path} missing base_model_name_or_path!")
            
        device_map = "auto" if torch.cuda.is_available() else None
        
        if load_in_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=True
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=device_map,
                trust_remote_code=True
            )
        
        logger.info(f"  Loading adapter...")
        model = PeftModel.from_pretrained(base_model, source_path)
    else:
        logger.info(f"  Loading full model...")
        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            source_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )
    
    tokenizer = fix_pad_token(tokenizer, model, context="load_model_transformers")
    return model, tokenizer

def load_bert_model(model_path, num_labels=3, id2label=None, label2id=None):
    """
    Load BERT for token classification with EXPLICIT label support.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        logger.info(f"  Detected PEFT adapter at: {model_path}")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        
        base_model_id = adapter_config.get('base_model_name_or_path')
        
        # Load base with explicit labels
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_id,
            num_labels=num_labels,
            id2label=id2label, 
            label2id=label2id
        )
        
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        logger.info(f"  Loading full model from: {model_path}")
        # Load full model with explicit labels
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            id2label=id2label, 
            label2id=label2id
        )
    
    return model, tokenizer