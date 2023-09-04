# -*- coding: utf-8 -*-
# @Author  : LG

from transformers import AutoTokenizer, AutoModel, TextStreamer


class LLM:
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(self.device)
        self.model = self.model.eval()

    def chat(self, question:str, history=[]):
        response, history = self.model.chat(self.tokenizer, question, history=history)
        return response, history

