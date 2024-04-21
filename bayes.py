import pickle
import datetime
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB


def output(actual, predict, output_file):
    """
    格式化输出预测结果
    :param actual: 真实类
    :param predict: 测试类
    :param output_file: 输出文件
    :return: 无返回值
    """
    precision = metrics.precision_score(actual, predict, average='micro')
    recall = metrics.recall_score(actual, predict, average='micro')
    f1_score = metrics.f1_score(actual, predict, average='micro')
    report = classification_report(actual, predict)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'f1-score: {f1_score:.4f}')
    print(report)

    with open(output_file, 'w') as file:
        file.write(f'Precision: {precision:.4f}\n')
        file.write(f'Recall: {recall:.4f}\n')
        file.write(f'f1-score: {f1_score:.4f}\n')
        file.write(report)


def bayes(class_list, bayes_output_file):
    """
    使用MultinomialNB分类器进行分类
    :param class_list: 类列表
    :param bayes_output_file:   输出文件路径
    :return: 无返回值
    """
    train_set_path = 'data/tf_idf_training'
    test_set_path = 'data/tf_idf_test'

    with open(train_set_path, 'rb') as file:
        train_bunch = pickle.load(file)

    with open(test_set_path, 'rb') as file:
        test_bunch = pickle.load(file)

    start_train_time = datetime.datetime.now()
    classifier = MultinomialNB(alpha=0.001).fit(train_bunch.tf_idf_weight_matrix, train_bunch.label)
    end_train_time = datetime.datetime.now()
    train_time = (end_train_time - start_train_time).microseconds / 1000

    start_test_time = datetime.datetime.now()
    predicted = classifier.predict(test_bunch.tf_idf_weight_matrix)
    end_test_time = datetime.datetime.now()
    test_time = (end_test_time - start_test_time).microseconds / 1000
    print(f'Forecast time is {test_time} millisecond.')

    output(test_bunch.label, predicted, bayes_output_file)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    confusion_matrix_df = pd.DataFrame(confusion_matrix(test_bunch.label, predicted), columns=class_list, index=class_list)
    print(confusion_matrix_df)
