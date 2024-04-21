import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import Bunch


def build_dict():
    """
    计算训练集的TF/IDF值，并构建词典，生成特征矩阵
    :return:
    """
    train_set_path = 'data/training'
    tf_idf_bunch_path = 'data/tf_idf_training'

    with open(train_set_path, 'rb') as file:
        train_set_bunch = pickle.load(file)

    start_time = datetime.datetime.now()

    vector = TfidfVectorizer(sublinear_tf=True, max_df=0.15, min_df=0.001, max_features=2500)

    tf_idf_bunch = Bunch(filenames=train_set_bunch.filenames, label=train_set_bunch.label, tf_idf_weight_matrix=[], vocabulary={})
    # 特征矩阵
    tf_idf_bunch.tf_idf_weight_matrix = vector.fit_transform(train_set_bunch.contents)

    # 输出词汇表和特征矩阵
    print(vector.get_feature_names_out())
    print(tf_idf_bunch.tf_idf_weight_matrix)

    end_time = datetime.datetime.now()
    duration_time = (end_time - start_time).seconds
    print(f'Calculate TF/IDF value using {duration_time} seconds.')

    tf_idf_bunch.vocabulary = vector.vocabulary_

    with open(tf_idf_bunch_path, 'wb') as file:
        pickle.dump(tf_idf_bunch, file)


def build_test_dict():
    """
    计算测试集的TF/IDF值，生成特征矩阵
    :return:
    """
    test_bunch_path = 'data/test'
    tf_idf_train_bunch_path = 'data/tf_idf_training'
    tf_idf_test_bunch_path = 'data/tf_idf_test'

    with open(test_bunch_path, 'rb') as file:
        test_bunch = pickle.load(file)

    with open(tf_idf_train_bunch_path, 'rb') as file:
        tf_idf_train_bunch = pickle.load(file)

    tf_idf_test_bunch = Bunch(filenames=test_bunch.filenames, label=test_bunch.label, tf_idf_weight_matrix=[], vocabulary=tf_idf_train_bunch.vocabulary)

    vector = TfidfVectorizer(sublinear_tf=True, max_df=0.15, min_df=0.001, vocabulary=tf_idf_train_bunch.vocabulary)
    tf_idf_test_bunch.tf_idf_weight_matrix = vector.fit_transform(test_bunch.contents)

    # print(vector.get_feature_names_out())
    # print(tf_idf_test_bunch.tf_idf_weight_matrix)

    with open(tf_idf_test_bunch_path, 'wb') as file:
        pickle.dump(tf_idf_test_bunch, file)
