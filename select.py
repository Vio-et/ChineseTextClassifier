import os


def count_words(file_path):
    """
    计算文件中的字数
    :param file_path: 待统计文件路径
    :return: 文件字数
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        word_count = len(content)
    return word_count


def select(input_folder, output_train_folder, output_test_folder, target_word_count):
    """
    将指定要求的文件从源目录转移至目标目录
    :param input_folder:    源目录
    :param output_train_folder: 训练集目标目录
    :param output_test_folder:  测试机训练目录
    :param target_word_count:   目标最佳文章长度
    :return:
    """
    if not os.path.exists(output_train_folder):
        os.makedirs(output_train_folder)

    if not os.path.exists(output_test_folder):
        os.makedirs(output_test_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    file_word_count = {}

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        word_count = count_words(file_path)
        file_word_count[file_name] = word_count

    sorted_files = sorted(file_word_count.items(), key=lambda x: abs(x[1] - target_word_count))
    selected_files_train = sorted_files[:data_size / 2]
    selected_files_test = sorted_files[data_size / 2: data_size]

    for file_name, _ in selected_files_train:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_train_folder, file_name)
        # print(source_path, destination_path)
        os.replace(source_path, destination_path)

    for file_name, _ in selected_files_test:
        source_path = os.path.join(input_folder, file_name)
        destination_path = os.path.join(output_test_folder, file_name)
        # print(source_path, destination_path)
        os.replace(source_path, destination_path)


if __name__ == "__main__":
    data_size = 5000
    class_list = ['currentPolitics', 'education', 'finance', 'game', 'houseProperty', 'recreation', 'society', 'sport', 'stock', 'technology']

    for class_name in class_list:
        input_folder_path = "../../Data/sourceData/" + class_name + '/' + class_name
        output_train_folder_path = "E:/College/Study/Grade_3/Autumn/AI/Data/Data/trainingSet/" + class_name
        output_test_folder_path = "E:/College/Study/Grade_3/Autumn/AI/Data/Data/testSet/" + class_name

        print(input_folder_path, output_train_folder_path, output_test_folder_path)

        target_word_num = 500  # Adjust this value based on your requirements
        select(input_folder_path, output_train_folder_path, output_test_folder_path, target_word_num)
