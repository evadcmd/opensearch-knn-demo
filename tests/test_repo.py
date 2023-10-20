import pytest

from knnsearch import repo
from knnsearch.model import Category

"""
@pytest.mark.asyncio
async def test_add():
    resp = await repo.save(Category(name="美容"))
    print(resp)
"""


# @pytest.mark.asyncio
# async def test_create_idx():
#     await repo.create_index()


@pytest.mark.asyncio
async def test_search():
    await repo.search("milk")
