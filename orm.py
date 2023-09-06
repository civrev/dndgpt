from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, SmallInteger, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from util import EMBEDDING_SIZE

SQLALCHEMY_DATABASE_URI = "postgresql+psycopg://admin:root@localhost:5438/admin"


def get_engine():
    return create_engine(SQLALCHEMY_DATABASE_URI, echo=True)


class Base(DeclarativeBase):
    pass


class Fruit(Base):
    __tablename__ = "fruits"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    page: Mapped[int] = mapped_column(SmallInteger)
    embedding = mapped_column(Vector(EMBEDDING_SIZE))


if __name__ == "__main__":
    with Session(get_engine()) as session:
        session.add(Fruit(name="kiwi"))
        session.commit()
