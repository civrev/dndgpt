"""
This program loads DnD 5e embeddings into postgres vector table
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from orm import Item, get_engine
from util import get_dnd_data, get_embedding_model

dnd = get_dnd_data()
embedder = get_embedding_model()

# split texts into reasonable chunks for vectorization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# break each page down into chunks and then make vectors from chunks
items = []
for ix, page in enumerate(dnd):
    print(f"Processing page text {ix}/{len(dnd)}")
    texts = text_splitter.create_documents([page["text"]])
    embeddings = embedder.encode(
        [x.page_content for x in texts], show_progress_bar=True
    )
    for embedding in embeddings:
        items.append(Item(page=page["page"], embedding=embedding))

# add the vectors to DB
engine = get_engine()
with Session(engine) as session:
    session.add_all(items)
    session.commit()
