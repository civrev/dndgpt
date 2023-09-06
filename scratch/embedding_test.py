import random

from util import get_embedding_model, get_generic_dataset

dataset = get_generic_dataset()
abstracts = dataset["abstract"]
random.shuffle(abstracts)

embedding_model = get_embedding_model()
embeddings = embedding_model.encode(abstracts[:100], show_progress_bar=True)

print(abstracts[0])
print(embeddings[0][:10], "...")
