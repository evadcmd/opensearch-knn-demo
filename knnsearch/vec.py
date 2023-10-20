# type: ignore
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-small")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def vectorize(*texts: str) -> list[list[float]]:
    batch_dict = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    embeddings = F.normalize(
        average_pool(
            model(**batch_dict).last_hidden_state, batch_dict["attention_mask"]
        ),
        p=2,
        dim=1,
    )
    return [embeddings[i].tolist() for i in range(len(texts))]


def of(text: str) -> list[float]:
    return vectorize(text)[0]
