# -*- coding: utf-8 -*-
# @Author  : LG

from sentence_transformers import SentenceTransformer
from typing import List
from numpy import array
from transformers import AutoModel, AutoTokenizer
import torch


class Embedding:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str or List[str]) -> array:
        encoded_input = self.tokenizer(text, padding=True, truncation=False, return_tensors='pt')
        for k, v in encoded_input.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                encoded_input[k] = v
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        result = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        if result.device != torch.device('cpu'):
            result = result.cpu()
        return result.numpy()

