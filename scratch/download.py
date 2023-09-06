"""
This script downloads:
* a sample dataset
* llama 2 LLM optimized by huggingface
"""
import os
import random

from datasets import load_dataset

dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]

# Extract abstracts to train on and corresponding titles
abstracts = dataset["abstract"]
titles = dataset["title"]


# log in for access to the model
# using personal hugging face token
from huggingface_hub import login

login(token=os.environ["HGF_TOKEN"])

# c
from torch import bfloat16, cuda

model_id = "meta-llama/Llama-2-7b-chat-hf"
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
print(device)

import transformers

# Quantization to load an LLM with less GPU memory
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16,  # Computation type
)

# Llama 2 Tokenizer
print("Loading tokenizer")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Llama 2 Model
# automatically cached to ~/.cache/huggingface/
print("Loading model")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()


# Our text generator
generator = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1,
)

prompt = "Could you explain to me how 4-bit quantization works as if I am 5?"
res = generator(prompt)
print(res[0]["generated_text"])
