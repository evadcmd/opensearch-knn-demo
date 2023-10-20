from pydantic import BaseModel


class Category(BaseModel):
    id: int | None = None
    name: str
    embedding: list[float] | None = None
