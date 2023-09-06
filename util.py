import gc
import json
import os

import transformers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch import bfloat16, cuda

here = os.path.dirname(os.path.abspath(__file__))
generic_dataset_name = "CShorten/ML-ArXiv-Papers"
EMBEDDING_SIZE = 384


def get_generic_dataset():
    """A generic dataset of mostly scientific/technical articles"""
    return load_dataset(generic_dataset_name)["train"]


def get_dnd_data():
    with open(os.path.join(here, "dnd_data.json")) as f:
        return json.load(f)


# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# this model doesn't work great, but need more than 8gb VRAM for the 13b model
model_id = "meta-llama/Llama-2-7b-chat-hf"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

# Quantization to load an LLM with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16,  # Computation type
)

# tokenizer to break text into tokens
# just loads the recommended tokenizer for the model
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)


def get_llm():
    """Base for the large language model

    massive data footprint - takes a long time to load
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return model


def get_embedding_model():
    """embedding model converts inputs into vectors

    https://huggingface.co/BAAI/bge-small-en

    Converts string inputs to array of 384 floats
    see EMBEDDING_SIZE
    """
    return SentenceTransformer("BAAI/bge-small-en")


def empty_vram(*objs):
    """delete objects and call garbage collector

    this is particularly useful if you need to free VRAM
    """
    for obj in objs:
        del obj
    # Empty VRAM
    gc.collect()
    gc.collect()


def get_chatbot(model):
    return transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1,
    )
