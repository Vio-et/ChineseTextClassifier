import os
import jieba
import jieba.posseg as ps


def preprocessing():
    """
    对源语料库进行分词和降噪，去停用词、去非名词
    :return: 无返回值
    """
    if not os.path.exists(training_set_path):
        os.makedirs(training_set_path)

    if not os.path.exists(test_set_path):
        os.makedirs(test_set_path)

    files = [f for f in os.listdir(data_training_set_path) if f.endswith('.txt')]

    for file_name in files:
        file_path = os.path.join(data_training_set_path, file_name)
        processed_words = ''

        with open(file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.read()
            words = ps.cut(lines, use_paddle=True)

            for word, flag in words:
                if flag == 'n' and word not in stop_words_list:
                    processed_words += (word + ' ')

        training_file_path = os.path.join(training_set_path, file_name)

        with open(training_file_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(processed_words)

    files = [f for f in os.listdir(data_test_set_path) if f.endswith('.txt')]

    for file_name in files:
        file_path = os.path.join(data_test_set_path, file_name)
        processed_words = ''

        with open(file_path, 'r', encoding='utf-8') as file_handle:
            lines = file_handle.read()
            words = ps.cut(lines, use_paddle=True)

            for word, flag in words:
                if flag == 'n' and word not in stop_words_list:
                    processed_words += (word + ' ')

        test_file_path = os.path.join(test_set_path, file_name)

        with open(test_file_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write(processed_words)


if __name__ == '__main__':
    """
    完成对数据的预处理，并转移值测试集和训练集目录
    """
    data_size = 5000
    class_name_list = ['currentPolitics', 'education', 'finance', 'game', 'houseProperty', 'recreation', 'society', 'sport', 'stock', 'technology']

    with open('src/stop_words_ch.txt', 'r') as file:
        stop_words_list = [i.strip() for i in file.readlines()]
    # print(stop_words_list)
    data_training_set_folder = 'E:/College/Study/Grade_3/Autumn/AI/Data/Data/trainingSet/'
    data_test_set_folder = 'E:/College/Study/Grade_3/Autumn/AI/Data/Data/testSet/'
    training_set_folder = 'trainingSet/'
    test_set_folder = 'testSet/'

    jieba.enable_paddle()

    for class_name in class_name_list:
        data_training_set_path = data_training_set_folder + class_name
        data_test_set_path = data_test_set_folder + class_name
        training_set_path = training_set_folder + class_name
        test_set_path = test_set_folder + class_name

        preprocessing()
