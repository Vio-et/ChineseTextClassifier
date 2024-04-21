from select import select
from buildDict import build_dict, build_test_dict
from svc import svc
from bayes import bayes

if __name__ == "__main__":
    # 类向量
    class_list = ['currentPolitics', 'education', 'finance', 'game', 'houseProperty', 'recreation', 'society', 'sport', 'stock', 'technology']

    # 类权重
    class_weights = {'currentPolitics': 0.5, 'education': 1, 'finance': 1, 'game': 1, 'houseProperty': 1, 'recreation': 1, 'society': 0.6, 'sport': 1, 'stock': 0.8, 'technology': 1}

    # 文件路径
    bayes_output_file = 'src/bayes.log'
    svm_output_file = 'src/svm.log'

    # 建立词典
    # build_dict()
    # build_test_dict()

    svc(class_list, class_weights, svm_output_file)   # SVM
    # bayes(class_list, bayes_output_file)                # 朴素贝叶斯
