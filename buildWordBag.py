import os
from sklearn.utils import Bunch
import pickle


def construct_word_bag():
    """
    构建词袋，格式化封装语料库
    :return: 无返回值
    """
    files = [f for f in os.listdir(training_set_path) if f.endswith('.txt')]

    for file_name in files:
        training_file_path = os.path.join(training_set_path, file_name)

        with open(training_file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.read()

        train_set.filenames.append(str(file_name))
        train_set.label.append(str(class_name))
        train_set.contents.append(str(lines))

    files = [f for f in os.listdir(test_set_path) if f.endswith('.txt')]

    for file_name in files:
        test_file_path = os.path.join(test_set_path, file_name)

        with open(test_file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.read()

        test_set.filenames.append(str(file_name))
        test_set.label.append(str(class_name))
        test_set.contents.append(str(lines))


if __name__ == '__main__':
    """
    构建测试集和训练集词袋
    """
    training_set_folder = 'trainingSet/'
    test_set_folder = 'testSet/'
    train_word_bag_path = 'data/training'
    test_word_bag_path = 'data/test'
    train_set = Bunch(label=[], filenames=[], contents=[])
    test_set = Bunch(label=[], filenames=[], contents=[])

    class_list = ['currentPolitics', 'education', 'finance', 'game', 'houseProperty', 'recreation', 'society', 'sport', 'stock', 'technology']

    for class_name in class_list:
        training_set_path = training_set_folder + class_name
        test_set_path = test_set_folder + class_name

        construct_word_bag()

    with open(train_word_bag_path, 'wb') as file:
        pickle.dump(train_set, file)

    with open(test_word_bag_path, 'wb') as file:
        pickle.dump(test_set, file)
