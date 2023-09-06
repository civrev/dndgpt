from sqlalchemy import select
from sqlalchemy.orm import Session

from orm import Item, get_engine
from prompts import DndBot
from util import get_embedding_model

dnd = DndBot()

q = "I am playing as a wizard. How many spells can I have at level 3?"
embedder = get_embedding_model()
engine = get_engine()


def get_relevant_pages(q):
    embedding = embedder.encode(q)
    with Session(engine) as session:
        # get 5 nearest neighbors by L2 distance
        rows = session.scalars(
            select(Item).order_by(Item.embedding.l2_distance(embedding)).limit(5)
        ).fetchall()
    return [r.page for r in rows]


# ask it the question with the context of the most relevant page
print(dnd.ask(q, get_relevant_pages[:1]))
