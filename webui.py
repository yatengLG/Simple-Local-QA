# -*- coding: utf-8 -*-
# @Author  : LG

from localQA import LocalQA, Embedding, LLM, VectorStore, Loader, logger
import gradio as gr
import os

knowledge_bases_root = 'knowledge_bases'  # 本地知识库存储根目录

def get_vector_store_list():
    vector_stores = os.listdir(knowledge_bases_root)
    return vector_stores

def change_mode(mode, chatbot):
    if mode == 'LLM对话':
        localQA.history = []
        return gr.update(visible=False), [[None, "进行LLM对话"]]
    elif mode == '知识库问答':
        localQA.history = []
        return gr.update(visible=True), [[None, "进行本地知识库问答"]]
    else:
        localQA.history = []
        return gr.update(visible=False), chatbot

def get_answer(mode, query, chatbot):
    if mode == 'LLM对话':
        # response = localQA.ask_llm(query)
        for response in localQA.stream_ask_llm(query):
            yield '', chatbot + [[query, response]]

    elif mode == '知识库问答':
        if localQA.vector_store is None:
            return '', chatbot+[[query, '请先选择知识库，然后进行知识库问答']]
        if localQA.vector_store.index_store.ntotal < 1:
            return '', chatbot+[[query, '知识库为空，请向知识库中添加知识后再进行提问']]

        chatbot = chatbot + [[query, '']]
        stream = localQA.stream_ask_kb(query)
        while True:
            try:
                response, docs, files = next(stream)
                yield '', chatbot[:-1] + [[query, response]]
            except Exception as e:
                print('e: ', e)
                message = '\n\n'
                for index, (doc, file) in enumerate(zip(docs, files)):
                    file_name = os.path.split(file)[-1]
                    message += """<details> <summary>材料 [{}] {}</summary>\n{}\n</details>\n\n""".format(index + 1,
                                                                                                        file_name, doc)
                yield '', chatbot[:-1] + [[query, response + message]]
                break
    else:
        response = '请选择正确的对话模式，而不是{}'.format(mode)
        return '', chatbot+[[query, response]]

def update_knowledeg_base():
    return gr.update(visible=True, choices=get_vector_store_list(), value=None)

def select_knowledeg_base(select_kb):
    if select_kb is None:
        localQA.setVectorStore(None)
        return gr.update(visible=True), gr.update(choices=[]), [[None, '未选择知识库']]

    select_kb_path = os.path.join(knowledge_bases_root, select_kb)
    if os.path.exists(select_kb_path):
        vector_store = VectorStore(select_kb_path, score_threshold=score_threshold)
        localQA.setVectorStore(vector_store)
        file_list = [os.path.split(f)[-1] for f in localQA.vector_store.file_store]
        return gr.update(visible=True), gr.update(choices=file_list), [[None, '选择知识库：{}'.format(select_kb)]]
    else:
        localQA.setVectorStore(None)
        return gr.update(visible=True, choices=get_vector_store_list(), value=None), gr.update(choices=[]), [[None, '知识库：{}不存在'.format(select_kb)]]


def creat_knowledeg_base(create_kb, chatbot):
    new_kb = os.path.join(knowledge_bases_root, create_kb)
    if os.path.exists(new_kb):
        return '', \
               gr.update(visible=True), \
               chatbot + [[None, '知识库已存在：{}'.format(create_kb)]]
    else:
        os.mkdir(new_kb)
        vector_store = VectorStore(new_kb, score_threshold=score_threshold)
        localQA.setVectorStore(vector_store)
        return '', \
               gr.update(visible=True, choices=get_vector_store_list(), value=create_kb), \
               [[None, '新建知识库：{}'.format(create_kb)]] + [[None, '选择知识库：{}'.format(create_kb)]]

def delete_root(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) > 0:
            for file in files:
                file = os.path.join(path, file)
                delete_root(file)
        os.rmdir((path))
    else:
        os.remove(path)

def delete_knowledeg_base(select_kb, chatbot):
    if select_kb is None:
        return gr.update(visible=True),chatbot + [[None, '请先选择要删除的知识库']]
    else:
        select_kb_path = os.path.join(knowledge_bases_root, select_kb)
        if os.path.exists(select_kb_path):
            try:
                delete_root(select_kb_path)
                localQA.setVectorStore(None)
                return gr.update(visible=True, choices=get_vector_store_list(), value=None), \
                       [[None, '已删除知识库：{}'.format(select_kb)]]
            except Exception as e:
                return gr.update(visible=True), \
                       [[None, '删除知识库时出错：{}'.format(e)]]

def add_file_to_knowledge_base(files, chatbot):
    if localQA.vector_store is None:
        return '', gr.update(choices=[]), chatbot + [[query, '未选择知识库']]
    if isinstance(files, list):
        for file in files:
            file = file.name
            localQA.add_file(file)
        file_list = [os.path.split(f)[-1] for f in localQA.vector_store.file_store]
        return None, gr.update(choices = file_list), chatbot + [[None, '添加完成，请开始提问']]

def delete_file_from_knowledge_base(delete_file, chatbot):
    localQA.delete_file(delete_file)
    file_list = [os.path.split(f)[-1] for f in localQA.vector_store.file_store]
    return gr.update(choices = file_list), chatbot + [[None, '删除完成']]

def clear_history():
    localQA.history = []
    return [[None, '重新开始对话']]


with gr.Blocks() as demo:
    with gr.Tab('对话'):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, '欢迎使用Simple Local QA']]).style(height=785)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入问题，按回车键提交")

            with gr.Column(scale=4):
                with gr.Blocks():
                    mode = gr.Radio(choices=['LLM对话', '知识库问答'], value='知识库问答', label='模式')
                    clear_history_button = gr.Button('清空历史聊天')

                kb_Accordion = gr.Accordion('知识库')
                with kb_Accordion:

                    update_kb = gr.Button('刷新知识库')

                    select_kb = gr.Dropdown(choices=get_vector_store_list(),
                                            label='选择知识库',
                                            interactive=True,
                                            value=None)

                    create_kb = gr.Textbox(placeholder='输入新知识库名称，回车提交',
                                           label='新建知识库')

                    delete_kb = gr.Button('删除知识库').style(container=True)


                    with gr.Tab("上传文件"):
                        add_file = gr.File(label='上传文件',
                                            file_types=['.pdf', '.txt'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                        add_file_button = gr.Button('上传并构建知识库')

                    with gr.Tab('上传文件夹'):
                        add_folder = gr.File(label='上传文件夹',
                                             file_count='directory',
                                             show_label=False
                                             )
                        add_folder_button = gr.Button('上传并构建知识库')

                    with gr.Tab('管理文件'):
                        delete_file = gr.CheckboxGroup(choices=[],
                                                       label='选择要删除的文件',
                                                       interactive=True)

                        delete_file_buton = gr.Button('删除选中文件，并重新构建知识库')

                # 事件
                query.submit(fn=get_answer,
                             inputs=[mode, query, chatbot],
                             outputs=[query, chatbot])

                mode.change(fn=change_mode,
                            inputs=[mode, chatbot],
                            outputs=[kb_Accordion, chatbot])

                clear_history_button.click(fn=clear_history,
                                           inputs=[],
                                           outputs=[chatbot])

                update_kb.click(fn=update_knowledeg_base,
                                inputs=[],
                                outputs=[select_kb])

                select_kb.change(fn=select_knowledeg_base,
                                 inputs=[select_kb],
                                 outputs=[select_kb, delete_file, chatbot])

                create_kb.submit(fn=creat_knowledeg_base,
                                 inputs=[create_kb, chatbot],
                                 outputs=[create_kb, select_kb, chatbot])

                delete_kb.click(fn=delete_knowledeg_base,
                                inputs=[select_kb, chatbot],
                                outputs=[select_kb, chatbot])

                add_file_button.click(fn=add_file_to_knowledge_base,
                                      inputs=[add_file, chatbot],
                                      outputs=[add_file, delete_file, chatbot],
                                      show_progress=True)

                add_folder_button.click(fn=add_file_to_knowledge_base,
                                        inputs=[add_folder, chatbot],
                                        outputs=[add_folder, delete_file, chatbot],
                                        show_progress=True)

                delete_file_buton.click(fn=delete_file_from_knowledge_base,
                                        inputs=[delete_file, chatbot],
                                        outputs=[delete_file, chatbot],
                                        show_progress=True)

    with gr.Tab('说明'):
        gr.Markdown("""
        # Simple Local QA
        欢迎使用Simple Local QA
        
        本项目是基于大模型的本地知识库问答系统的简易版实现，但具有本地知识问答所需的所有功能。

        ## Web UI功能
        
        1. 提供了[LLM对话]与[知识库问答]两种模式。
        2. 支持新建知识库、删除知识库、向知识库中添加知识、从知识库中删除特定文件的知识等功能。
        
        ## 注意
        1. 当前知识库支持txt、pdf文件，可手动在loader中添加对其他文件的支持。
        2. 对pdf文件的支持并不完美，这是由pdf的存储顺序引起的。
        3. 大模型为chatglm2-6b-int4，Embedding为text2vec-large-chinese是因为这两个模型对中文支持比较好，且显存要求低。你可以替换任何其他模型。
        """)


if __name__ == '__main__':
    llm_path = './checkpoints/chatglm2-6b-int4'
    llm_device = 'cuda:0'

    embedding_path = './checkpoints/text2vec-large-chinese'
    embedding_device = 'cuda:1'

    score_threshold = 1000   # 知识库检索分数阈值
    sentence_size = 100     # 知识库文本

    embedding = Embedding(embedding_path, device=embedding_device)
    llm = LLM(llm_path, device=llm_device)
    vector_store = None
    loader = Loader(sentence_size)

    localQA = LocalQA(llm=llm,
                      embedding=embedding,
                      loader=loader,
                      vector_store=vector_store,
                      logger=logger)

    demo.queue(concurrency_count=3).launch()

