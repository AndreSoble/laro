import os
import traceback
from pprint import pprint
from typing import *
import collections

import torch
from torch import FloatTensor, cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from modeling_laro import LARO
from train_utils import device


@torch.no_grad()
def embed_labse(sentences: List[str]):
    print("Embedding using LABSE")
    bs_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 40))
    batches = [sentences[i:(i + bs_size)] for i in range(0, len(sentences), bs_size)]
    all_embeddings = list()
    for batch in batches:
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt').to(
            device)
        model_output = model(**encoded_input)
        embeddings = model_output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings)
    return all_embeddings


@torch.no_grad()
def embed_laro(sentences: List[str]):
    bs_size = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 40))
    batches = [sentences[i:(i + bs_size)] for i in range(0, len(sentences), bs_size)]
    print("Embedding using LARO")
    all_embeddings = list()
    for batch in batches:
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=64, return_tensors='pt').to(
            device)
        embeddings = model.get_embedding(input_ids=encoded_input["input_ids"],
                                         attention_mask=encoded_input["attention_mask"])
        all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings)
    return all_embeddings


@torch.no_grad()
def tp_fp_eval(embedding: FloatTensor, embeddings: FloatTensor, index: int) -> str:
    """
    :param sentence:
    :param sentences:
    :param index:
    :return: "tp" or "fp"
    """
    sim = cosine_similarity(torch.cat([embedding.unsqueeze(0) for _ in range(embeddings.size(0))]), embeddings)
    value, indices = torch.topk(sim, 1)
    if indices[0] == index:
        return "tp"
    else:
        return "fp"


def evaluate(model):
    """
    :param model: "labse" or "laro"
    :return:
    """
    model.eval()
    files = os.listdir("./v1")

    lang_counter = dict()

    for i in tqdm(range(0, len(files), 2)):
        print("")
        print(f"Starting with {files[i], files[i + 1]}")
        source_sentences = open("./v1/" + files[i], encoding="utf8").readlines()
        target_sentences = open("./v1/" + files[i + 1], encoding="utf8").readlines()
        counter = Counter()
        source_embeddings = embed_labse(source_sentences) if model == "labse" else embed_laro(source_sentences)
        target_embeddings = embed_labse(target_sentences) if model == "labse" else embed_laro(target_sentences)
        for index, sentence in enumerate(source_sentences):
            counter[tp_fp_eval(source_embeddings[index], target_embeddings, index)] += 1
        lang_counter[files[i].split(".")[1]] = counter
        print("")
        pprint(lang_counter)
    print("#" * 100)
    pprint(lang_counter)


if __name__ == "__main__":
    checkpoints = os.listdir(os.environ.get("OUTPUT_DIR", './results'))
    n = ""
    x = 0.0
    for c in checkpoints:
        m = c.split("-")[1]
        if float(m) > x:
            n = c
            x = float(m)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = LARO.from_pretrained(os.environ.get("OUTPUT_DIR", './results') + "/" + str(n)).to(device)
    model.eval()
    evaluate("laro")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)
    model.eval()
    evaluate("labse")
