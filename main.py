# -*- coding: utf-8 -*-
# @Author  : LG

from localQA import LocalQA, Embedding, LLM, VectorStore, Loader, logger

embedding = Embedding('/mnt/disk2/text2vec-large-chinese')
llm = LLM('/mnt/disk2/chatglm2-6b-int4')
vector_store = VectorStore('/mnt/disk2/PycharmProjects/langchain学习/knowledge_bases/测试')
loader = Loader(100)


localQA = LocalQA(llm=llm,
                  embedding=embedding,
                  loader=loader,
                  vector_store=vector_store,
                  logger=logger)

response = localQA.ask_kb('核心点卷积')
print(response)