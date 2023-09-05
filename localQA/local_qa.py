# -*- coding: utf-8 -*-
# @Author  : LG

from localQA.embedding import Embedding
from localQA.llm import LLM
from localQA.vectorstore import VectorStore
from localQA.loader import Loader
import torch
import shutil
import os


class LocalQA:
    def __init__(self, llm, embedding:Embedding=None, loader:Loader=None, vector_store:VectorStore=None, logger=None):
        self.llm = llm
        self.embedding = embedding
        self.loader = loader
        self.vector_store = vector_store
        self.logger = logger
        self.prompt_template = u"""已知信息：{context}\n\n根据上述已知信息，专业的回答用户提出的问题。如果无法从中得到答案，请说“根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。问题是：{question} """
        self.history = []

        self.init_logger()
        if self.llm is not None:
            self.logger.info(u'Init [{:^14s}] from {}'.format('LLM', self.llm.model_path))
        if self.embedding is not None:
            self.logger.info(u'Init [{:^14s}] from {}'.format('Embedding', self.embedding.model_path))
        if self.vector_store is not None:
            self.logger.info(u'Init [{:^14s}] from {}'.format('Vector Store', self.vector_store.root))

        if self.vector_store is not None:
            if not os.path.exists(os.path.join(self.vector_store.root, 'tmps')):
                os.mkdir(os.path.join(self.vector_store.root, 'tmps'))
            if not os.path.exists(os.path.join(self.vector_store.root, 'docs')):
                os.mkdir(os.path.join(self.vector_store.root, 'docs'))

    def init_logger(self):
        if self.logger is None:
            try:
                from logging import getLogger
                self.logger = getLogger('')
            except:
                raise ImportError(u"Can't import logging.")

    def add_file(self, file_path, save_tmp=True):
        self.logger.info(u'Add file: {}'.format(file_path))
        self.logger.info(u'Loading file...')
        texts = self.loader.load(file_path)
        file_name = os.path.split(file_path)[-1]
        if not texts:
            self.logger.info(u'Empty file.')
            return
        if save_tmp:
            self.logger.info(u'Writing tmp...')
            tmp_path = os.path.join(self.vector_store.root, 'tmps', '.'.join(file_name.split('.')[:-1]) + '.txt')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                for i, text in enumerate(texts):
                    f.write(u'{}\n'.format(text))
        # 分批，不然显存占用太大了
        split = 500
        for i in range(len(texts)//split+1):
            self.logger.info(u'Embeding text...')
            split_texts = texts[int(i*split): int((i+1)*split)]
            embedings = self.embedding.embed_text(split_texts)
            self.logger.info(u'Writing vector store...')
            self.vector_store.add_data(split_texts, embedings, file_name)
        self.vector_store.save()
        self.empty_cuda()

    def delete_file(self, files_name: str or list):
        self.logger.info(u'Deleting [{}] ...'.format(files_name))
        if isinstance(files_name, str):
            files_name = [files_name]
        result, message = self.vector_store.delete_by_file_name(files_name)
        if result:
            self.logger.error(u'Delete success.')
        else:
            self.logger.error(u'Delete error: {}'.format(result))
        self.empty_cuda()

    def search(self, question:str):
        self.logger.info(u"Searching '{}' from vector store...".format(question))
        embeding = self.embedding.embed_text(question)
        docs, files = self.vector_store.search(embeding)
        for doc, file in zip(docs, files):
            self.logger.info(u"Search result: 《{}》\t{}".format(file, doc))
        self.empty_cuda()
        return docs, files

    def ask_kb(self, question):
        docs, files = self.search(question)
        context = '\n'.join(docs)
        prompt = self.prompt_template.format(context=context, question=question)
        response, history = self.llm.chat(prompt, history=self.history)
        self.history = history
        self.empty_cuda()
        return response, docs, files

    def ask_llm(self, question):
        response, history = self.llm.chat(question, history=self.history)
        self.history = history
        self.empty_cuda()
        return response

    def stream_ask_kb(self, question):
        docs, files = self.search(question)
        context = '\n'.join(docs)
        prompt = self.prompt_template.format(context=context, question=question)
        for response, history in self.llm.stream_chat(prompt, history=self.history):
            self.history = history
            self.empty_cuda()
            yield response, docs, files

    def stream_ask_llm(self, question):
        for response, history in self.llm.stream_chat(question, history=self.history):
            self.history = history
            self.empty_cuda()
            yield response

    def clear(self):
        self.vector_store.clear()

    def empty_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def setLLM(self, llm):
        self.llm = llm

    def setEmbedding(self, embedding):
        self.embedding = embedding

    def setVectorStore(self, vector_store):
        self.vector_store = vector_store
        if self.vector_store is not None:
            if not os.path.exists(os.path.join(self.vector_store.root, 'tmps')):
                os.mkdir(os.path.join(self.vector_store.root, 'tmps'))
            if not os.path.exists(os.path.join(self.vector_store.root, 'docs')):
                os.mkdir(os.path.join(self.vector_store.root, 'docs'))

