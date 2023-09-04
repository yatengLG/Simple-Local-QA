# -*- coding: utf-8 -*-
# @Author  : LG

import paddleocr
import re
from typing import List
import numpy as np

class Loader:
    def __init__(self, sentence_size):
        self.sentence_size = sentence_size

    def load(self, file:str)-> List[str]:
        if file.lower().endswith('.txt'):
            text = self._load_txt(file)
        elif file.lower().endswith('.pdf'):
            text = self._load_pdf(file, keep_img=False)
        else:
            raise ValueError
        texts = self._split_text(text)
        return texts
        # return text

    def _load_txt(self, file)->str:
        try:
            with open(file) as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError("Error while loading file {}: {}".format(file, e))
        return text

    def _load_pdf(self, pdf_path, keep_img=False)->str:
        import fitz
        text = ''

        doc = fitz.open(pdf_path)
        for index in range(doc.page_count):
            page = doc[index]
            content = page.get_text('text')

            # 删除与中文连接的空格,除非另一端是换行号
            content = re.sub('(?<=[\u4e00-\u9fa5]) (?=[^\n])|(?<=[^\n]) (?=[\u4e00-\u9fa5])', '', content)
            # 删除一般回车，除非以空格结尾.
            content = re.sub("(?<=[^ 。；？！）.;?!)a-zA-Z])\n", '', content)
            # 删除英文　\n中的回车
            content = re.sub("(?<=[a-zA-Z] )\n", '', content)
            # 英文　\n中的回车
            content = re.sub("(?<=[a-zA-Z])\n", ' ', content)
            text += content

        # 将图片单独放在最后，不打断文本顺序
        if keep_img:
            ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            for index in range(doc.page_count):
                page = doc[index]
                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(doc, img[0])
                    img_content = ocr.ocr(pix.tobytes())
                    img_content = [r[1][0] for line in img_content for r in line]
                    img_content = '\n'.join(img_content)
                    text += img_content
        return text

    def _split_text(self, text)->List[str]:
        result = []
        # 一行基本就是一段
        for line in text.split('\n'):
            line_result = []
            line = line.strip(' ')
            if len(line) < 1:
                continue
            if len(line) < self.sentence_size:
                line_result.append(line)
            else:
                # 开始分割本行
                content_list = [line]
                # 找出引号“”""中文字，作为一个整体，方便后续按句号逗号分割句子
                while True:
                    content = content_list.pop(-1)
                    res = re.search('([“"][^”"]*[”"][?.。,，;；!！ ])', content)
                    if res is not None:
                        start, end = res.start(), res.end()
                        content_list.append(content[:start])
                        content_list.append(res.group())
                        content_list.append(content[end:])
                    else:
                        content_list.append(content)
                        break

                for content in content_list:
                    if len(content) > self.sentence_size:
                        # 。分割
                        sep = ['。', '；', '，', ' ']
                        res = re.search('[\u4e00-\u9fa5]', content)
                        if res is None:
                            sep = ['.', ';', ',', ' ']
                        content_split1 = self._split_with_sep(content, sep[0])
                        for cs1_index, cs1 in enumerate(content_split1):
                            if len(cs1) > self.sentence_size:
                                # ，分割
                                content_split2 = self._split_with_sep(cs1, sep[1])
                                for cs2_index, cs2 in enumerate(content_split2):
                                    if len(cs2) > self.sentence_size:
                                        content_split3 = self._split_with_sep(cs2, sep[2])
                                        for cs3_index, cs3 in enumerate(content_split3):
                                            if len(cs3) > self.sentence_size:
                                                content_split4 = self._split_with_sep(cs3, sep[3])
                                                for cs4_index, cs4 in enumerate(content_split4):
                                                    if len(cs4) > self.sentence_size:
                                                        # 按长度截取
                                                        while cs4:
                                                            line_result.append(cs4[:self.sentence_size])
                                                            cs4 = cs4[self.sentence_size:]
                                                    else:
                                                        line_result.append(cs4)
                                            else:
                                                line_result.append(cs3)
                                    else:
                                        line_result.append(cs2)
                            else:
                                line_result.append(cs1)
                    else:
                        line_result.append(content)
                # # 合并小句子
                # index = 0
                # while True:
                #     if index >= len(line_result):
                #         break
                #     if index + 1 < len(line_result):
                #         if len(line_result[index]) + len(line_result[index + 1]) < self.sentence_size:
                #             line_result.insert(index, line_result.pop(index) + line_result.pop(index))
                #         else:
                #             index += 1
                #     else:
                #         break
            result.extend(line_result)

        return result

    def _split_with_sep(self, text, sep):
        content_split = text.split(sep)
        # 添加句号，保持文本不变
        content_split = [c + sep for c in content_split[:-1]] + [content_split[-1]]
        # 合并小句子
        index = 0
        while True:
            if index >= len(content_split):
                break
            if index + 1 < len(content_split):
                if len(content_split[index]) + len(content_split[index + 1]) < self.sentence_size:
                    content_split.insert(index,
                                         content_split.pop(index) + content_split.pop(index))
                else:
                    index += 1
            else:
                break
        return content_split

