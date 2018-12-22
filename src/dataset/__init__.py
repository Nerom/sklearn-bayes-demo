# encoding=utf-8
import jieba
import os
import re
import logging
import sys
from sklearn.feature_extraction.text import CountVectorizer

# 日志设置
console = logging.StreamHandler(sys.stderr)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(console)


class DataSet:
    __stop_word_list = []
    __train_data = []
    __train_vector = None
    __train_tag_arr = []
    __test_data = []
    __test_vector = None
    __test_tag_arr = []
    __test_file_name = []
    __vectorizer = None

    def __init__(self):
        jieba.default_logger.setLevel(logging.INFO)
        self.__test_res_arr = []

    @staticmethod
    def _load():
        # 向量化工具对象
        log.info("initializing vectorizer")
        DataSet.__vectorizer = CountVectorizer(analyzer=lambda s: DataSet.cut(s))
        log.info("vectorizer initialized")
        # 分别加载正常邮件、垃圾邮件及测试文件
        log.info("loading data from files")
        DataSet.__load_stop_words()
        DataSet.__load_train_data()
        DataSet.__load_test_data()
        log.info("data from files loaded")
        # 向量化训练数据和测试数据
        log.info("train data start vectorization")
        DataSet.__train_vector = DataSet.__vectorizer.fit_transform(DataSet.__train_data).toarray()
        log.info("train data vectorized")
        log.info("test data start vectorization")
        DataSet.__test_vector = DataSet.__vectorizer.transform(DataSet.__test_data).toarray()
        log.info("test data vectorized")

    def get_train_data(self):
        return self.__train_vector

    def get_tag(self):
        return self.__tag_data

    def get_test_data(self):
        return self.__test_vector

    def load_predict_result(self, pre_result):
        self.__test_res_arr = pre_result

    def print_detail(self):
        for i in range(len(self.__test_tag_arr)):
            if self.__test_tag_arr[i] != self.__test_res_arr[i]:
                print("[DETAIL] false predicate, file:[%s], content:[%s]" %
                      (self.__test_file_name[i], self.__test_data[i]))
        self.print_summary()

    def print_summary(self):
        t = 0
        f = 0
        for i in range(len(self.__test_tag_arr)):
            if self.__test_tag_arr[i] == self.__test_res_arr[i]:
                t += 1
            else:
                f += 1
        print("[SUMMARY] accuracy: %.2f\n" % (t / (t + f) * 100))

    @staticmethod
    def cut(content):
        """
        获取文本词语列表及各个词语出现的次数
        :param content:
        :return:
        """
        words = []
        # 分词结果放入res_list
        word_list = list(jieba.cut(content))
        for word in word_list:
            if word not in DataSet.__stop_word_list and word.strip() != '' and word is not None:
                words.append(word)
        return words

    @staticmethod
    def __load_stop_words():
        # 获得停用词表
        for line in open("./dataset/data/zh-stop-words.txt", encoding="gbk"):
            DataSet.__stop_word_list.append(line[:len(line) - 1])

    @staticmethod
    def __load_train_data():
        """
        加载所有有标签文件
        :return:
        """
        norm_file_list = DataSet.__get_file_list(r"./dataset/data/normal")
        spam_file_list = DataSet.__get_file_list(r"./dataset/data/spam")
        # spam在前  normal在后
        DataSet.__train_data = DataSet.__get_content_from_file(spam_file_list) + DataSet.__get_content_from_file(
            norm_file_list)
        DataSet.__tag_data = [1 for n in range(len(spam_file_list))] + [0 for n in range(len(norm_file_list))]

    @staticmethod
    def __load_test_data():
        """
        加载测试数据
        :return:
        """
        test_file_list = DataSet.__get_file_list(r"./dataset/data/test")
        DataSet.__test_file_name = test_file_list
        DataSet.__test_data = DataSet.__get_content_from_file(test_file_list)
        # 文件名小于1000的是正常邮件，大于1000的是垃圾邮件
        for file_name in test_file_list:
            str_list = file_name.split(r"/")
            file_i = int(str_list[len(str_list) - 1])
            if file_i < 1000:
                DataSet.__test_tag_arr.append(0)
            else:
                DataSet.__test_tag_arr.append(1)

    @staticmethod
    def __get_content_from_file(file_list):
        """
        抽取样本文件的关键词及其出现次数
        :param file_list:
        :return:
        """
        words_list = []
        # 获得垃圾邮件中的词频
        for file_name in file_list:
            content = ""
            for line in open(file_name, encoding="gbk"):
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                content += line
            words_list.append(content)
        return words_list

    def __get_word_dic(self, line):
        """
        获取文本词语列表及各个词语出现的次数
        :param line:
        :return:
        """
        word_dic = {}
        # 分词结果放入res_list
        word_list = list(jieba.cut(line))
        for word in word_list:
            if word not in self.__stop_word_list and word.strip() != '' and word is not None:
                if word not in word_dic.keys():
                    word_dic.setdefault(word, 1)
                else:
                    word_dic[word] += 1

    @staticmethod
    def __get_file_list(file_path):
        file_name_list = os.listdir(file_path)
        result = list(map(lambda file_name:
                          file_path + "" + file_name
                          if file_name[len(file_name) - 1] == "/"
                          else file_path + "/" + file_name, file_name_list))
        return result


# 初始化加载数据
log.info("initializing dataset")
DataSet._load()
log.info("dataset initialized")
