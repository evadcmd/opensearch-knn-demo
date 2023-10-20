from opensearchpy import AsyncOpenSearch

from knnsearch import vec
from knnsearch.model import Category

opensearch = AsyncOpenSearch(hosts=["http://localhost:9200"])


async def create_index():
    await opensearch.indices.create(
        index="categories",
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "text"},
                    "embedding": {"type": "knn_vector", "dimension": 384},
                }
            },
        },
    )


async def save(category: Category):
    category.embedding = vec.of(category.name)
    await opensearch.index(index="categories", body=category.model_dump())


async def search(text: str, k: int = 1):
    results = await opensearch.search(
        index="categories",
        body={
            "query": {
                "knn": {
                    "embedding": {
                        "vector": vec.of(text),
                        "k": 2,
                    }
                },
            },
        },
    )
    for hit in results["hits"]["hits"][:2]:
        print(hit["_source"]["name"])
