# encoding=utf-8
"""
Created on 2016年4月18日

@author: lenovo
"""

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from src.dataset import DataSet

# 贝叶斯分类器对象
for classifier in [BernoulliNB(), MultinomialNB(), GaussianNB()]:
    print("classifier: [%s]" % type(classifier).__name__)
    # 数据类对象
    data = DataSet()

    # 获取带标签训练数据
    train_X = data.get_train_data()
    train_Y = data.get_tag()

    # 训练
    print("start training")
    classifier.fit(train_X, train_Y)
    print("training done")

    # 获取向量化测试数据
    test_X = data.get_test_data()
    # 预测结果
    print("start predicating")
    result = classifier.predict(test_X)
    print("predicate done")

    # 加载回数据对象，用来输出结果用
    print("collecting result")
    data.load_predict_result(list(result))

    # 输出明细
    data.print_detail()
