# -*- coding: utf-8 -*-
# @Author  : LG

import logging

logging.getLogger()

class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super(Logger, self).__init__(name, level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler("log.txt")  # 文件处理器
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        console = logging.StreamHandler()  # 控制台处理器
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        self.addHandler(handler)  # 添加文件输出
        self.addHandler(console)  # 添加控制台输出

logger = Logger('Simple Local QA')

