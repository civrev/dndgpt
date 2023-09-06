# dndgpt
A document oriented agent that helps with dnd questions

## context/design
this project is based around designing a `Document Oriented Agent` that helps with dnd questions

A `Document Oriented Agent` is a language model that assists with specialized documentation.

This is not LLM *fine-tuning* which is training an LLM on example desired conversations
(though this option seems like a good way to improve the model after user testing).

A good design guide to the kind of agent this project is about can be found [here](https://bdtechtalks.com/2023/05/01/customize-chatgpt-llm-embeddings/).

The approach is that specialized documentation is fed through a tokenizer that turns
all the documentation into vectors - this is the *embedding model*

# conda environment
make sure you use the `mamba` solver. It is way faster.

create the conda environment then use pip
```
conda env create -f environment.yml
pip install -r requirements.txt
```

## GPU help
if you have a Nvidia GPU with at least 8gb of VRAM, you can probably use it to help out with this project.

## specialized data
I am using a copy of DnD 5th edition manual as the specialized data give context to the LLM.

This is probably not the best data to use to this purpose as it is too technical.

I searched `dnd 5e pdf` and took the first result ~293 pages with selectable text.

Pages were trimmed to exclude table of contents, preface, index, etc.
```
import json
from PyPDF2 import PdfReader
reader = PdfReader('dnd5e.pdf')
pages_w_count = list(enumerate(reader.pages))
data = [{'page': ix, 'text': p.extract_text()} for ix, p in pages_w_count[3:289]]
with open("dnd_data.json", "w") as f:
    json.dump(data, f)
```