# -*- coding: utf-8 -*-
# @Author  : LG

import faiss
import pickle
import os
from typing import List
import numpy as np
import copy

class VectorStore:
    def __init__(self, root:str, dim=1024, score_threshold=1000):
        self.root = root
        self.dim = dim
        self.score_threshold = score_threshold

        if not os.path.exists(root):
            os.mkdir(root)
        self.index_store_path = os.path.join(root, 'index.faiss')
        self.doc_store_path = os.path.join(root, 'doc.pkl')
        self.file_store_path = os.path.join(root, 'file.pkl')

        self.index_store = faiss.read_index(self.index_store_path) \
            if os.path.exists(self.index_store_path) else faiss.IndexFlatL2(dim)
        if os.path.exists(self.doc_store_path):
            with open(self.doc_store_path, 'rb') as f:
                self.doc_store = pickle.load(f)
        else:
            self.doc_store = []
        if os.path.exists(self.file_store_path):
            with open(self.file_store_path, 'rb') as f:
                self.file_store = pickle.load(f)
        else:
            self.file_store = set()

    def add_data(self, texts:List[str], embedings, file_name):
        file_store = copy.copy(self.file_store)
        doc_store = copy.copy(self.doc_store)
        index_store = copy.copy(self.index_store)
        try:
            for index, text in enumerate(texts):
                self.doc_store.append({'metadata':file_name, 'page_content': text})
            self.index_store.add(embedings)
            self.file_store.add(file_name)
            return True, '添加数据成功,添加后向量数量为{}'.format(self.index_store.ntotal)

        except Exception as e:
            self.file_store = file_store
            self.doc_store = doc_store
            self.index_store = index_store
            return False, e

    def exist_file(self, file_name:str):
        return file_name in self.file_store

    def delete_by_file_name(self, files_name:List[str]):
        ids = [index for index, d in enumerate(self.doc_store) if d.get('metadata', '') in files_name]
        ids = sorted(ids, reverse=True)
        file_store = copy.copy(self.file_store)
        doc_store = copy.copy(self.doc_store)
        index_store = copy.copy(self.index_store)
        try:
            self.index_store.remove_ids(np.array(ids))
            for id in ids:
                self.doc_store.pop(id)
            for file_name in files_name:
                self.file_store.remove(file_name)
            self.save()
            return True, '删除数据成功，删除后向量数量为{}'.format(self.index_store.ntotal)
        except Exception as e:
            self.file_store = file_store
            self.doc_store = doc_store
            self.index_store = index_store
            return False, e

    def search(self, embeding, k:int=5):
        if embeding.ndim == 1:
            embeding = embeding[np.newaxis, :]
        scores, indices = self.index_store.search(embeding, k)
        scores, indices = scores[0], indices[0]

        scores_mask = scores < self.score_threshold
        indices = indices[scores_mask]
        indices.sort()

        indices_groups = []
        i = 0
        while i < len(indices):
            group = []
            group.append(indices[i])
            while i+1 < len(indices) and indices[i+1]-indices[i] <= 2 and self.doc_store[indices[i]].get('metadata', '') == self.doc_store[indices[i+1]].get('metadata', ''):
                group.append(indices[i+1])
                i += 1
            i+=1
            indices_groups.append(group)
        docs = []
        file_lists = []
        for group in indices_groups:
            doc = ''
            start_ind, end_ind = group[0], group[-1]
            for ind in range(start_ind-3, end_ind+4):
                if 0<= ind < len(self.doc_store):
                    # if self.doc_store[ind].get('metadata', '') == self.doc_store[index].get('metadata', ''):
                    doc += self.doc_store[ind].get('page_content', '')
            docs.append(doc)
            file_lists.append(self.doc_store[start_ind].get('metadata', ''))
        # context = '\n'.join(docs)
        return docs, file_lists

    def clear(self):
        file_store = copy.copy(self.file_store)
        doc_store = copy.copy(self.doc_store)
        index_store = copy.copy(self.index_store)
        try:
            self.index_store.reset()
            self.doc_store.clear()
            self.file_store.clear()
            self.save()
            return True, '知识库清空成功'
        except Exception as e:
            self.file_store = file_store
            self.doc_store = doc_store
            self.index_store = index_store
            return False, e

    def save(self):
        try:
            faiss.write_index(self.index_store, self.index_store_path)
            with open(self.doc_store_path, 'wb') as f:
                pickle.dump(self.doc_store, f)
            with open(self.file_store_path, 'wb') as f:
                pickle.dump(self.file_store, f)
            return True, '知识库保存成功'
        except Exception as e:
            return False, e

