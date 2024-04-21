import datetime
import pickle
import numpy as np
import threading

train_set_path = 'data/tf_idf_training'
test_set_path = 'data/tf_idf_test'

with open(train_set_path, 'rb') as file:
    train_bunch = pickle.load(file)

with open(test_set_path, 'rb') as file:
    test_bunch = pickle.load(file)

D = 2000    # 维度
m = train_bunch.tf_idf_weight_matrix.toarray()
n = test_bunch.tf_idf_weight_matrix.toarray()
P = np.zeros((D, 10))

out_p = np.zeros((50000, 2))
out_s = np.zeros((((((((((10, 10))))))))))

threadLock = threading.Lock()
threads = []


def get_pos(list_data, max_data):
    max_loc = 0
    tuple_data = tuple(list_data)

    for index in range(len(tuple_data)):
        if tuple_data[index] == max_data:
            max_loc = index
            break

    return max_loc


class MyThread(threading.Thread):
    def __init__(self, begin_pos):
        threading.Thread.__init__(self)
        self.begin_pos = begin_pos

    def run(self):
        for word in range(self.begin_pos, self.begin_pos + 5000):
            # print(f'Predicting {word} pages')
            s = np.zeros(10)
            for i in range(0, D):
                for j in range(0, 10):
                    s[j] += np.log(n[word][i] * P[i][j] + 1)
            pos = get_pos(s, (np.max(s)))
            threadLock.acquire()
            out_p[word][0] = int(word / 5000)
            out_p[word][1] = int(pos)
            out_s[int(word / 5000)][int(pos)] += 1
            threadLock.release()
            # print(f'Predict class {int(pos)}, actual class {int(word / 5000)}')


def init():
    for j in range(D):
        tot_sum = 0.0
        type_sum = np.zeros(10)
        for i in range(50000):
            current_type = int(i / 5000)
            current_value = m[i][j]
            type_sum[current_type] += current_value
            tot_sum += current_value
        for k in range(10):
            P[j][k] = type_sum[k] / tot_sum
            # print(f'The rate of words of class {j} in {k} is {P[j][k]}')


class_list = ['currentPolitics', 'education', 'finance', 'game', 'houseProperty', 'recreation', 'society', 'sport', 'stock', 'technology']
start_time = datetime.datetime.now()

init()

for i in range(10):
    threads.append(MyThread(5000 * i))
    threads[i].start()
for i in range(10):
    threads[i].join()

end_time = datetime.datetime.now()
duration_time = (end_time - start_time).seconds
print(f'Own bayes execution time is {duration_time} seconds.')

for i in range(10):
    print(class_list[i]+'   ', end='')

for i in range(10):
    print(class_list[i]+'   ', end='')
    for j in range(10):
        print(str(int(out_s[i][j]))+' ', end='')
    print('     ')

success_num = 0
for i in range(10):
    success_num += out_s[i][i]

print(success_num)

total = 0
total_recall = 0
f1 = 0

for i in range(10):
    recall = out_s[i][i] / 5000
    total_recall += recall
    total_sum = 0

    for j in range(0, 10):
        total_sum += out_s[j][i]

    sp = out_s[i][i] / total_sum
    total += sp
    f = (2.0 * recall * sp) / (sp + recall)
    f1 += f
    print(class_list[i] + "Class Precision：" + str(sp) + "，Recall：" + str(recall) + "，f1-score 为：" + str(f))

total_recall = (total_recall * 1.0) / 10
total = (total * 1.0) / 10
f1 = (f1 * 1.0) / 10

print("Precision：" + str(total))
print("Recall：" + str(total_recall))
print("f1-score = " + str(f1))
