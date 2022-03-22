import numpy as np
import pandas as pd
import copy
import pickle as pkl
from numpy.linalg import norm

class FunkSVD():

    def __init__(self, true_train_dataset, validation_dataset, user_embeddings, item_embeddings,  n_latent, max_epochs, lr, initialization_method='normal'):
        self.true_train_dataset = true_train_dataset
        self.validation_dataset = validation_dataset
        self.user_embeddings = user_embeddings                                                                          # 用字典来存储User和item的嵌入向量。这里首先对数据结构类型进行声明
        self.item_embeddings = item_embeddings
        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.lr = lr
        self.initalization_method = initialization_method

    def train(self, early_stopping):                                                                                    # initialization_method是规定生成初始嵌入向量的方法，random或者normal，如果是normal则需要传入后面的mean和std参数，毕竟是高斯分布嘛
        self.initialize_latent_vectors()                                                                                # 到这就可以打印初始嵌入向量了，即 print(self.user_features,self.item_features)
        self.min_validation_error = np.inf
        error_counter = 0
        best_epoch = 0
        for _ in range(self.max_epochs):
            for row in self.true_train_dataset.iloc():
                error = row[2] - np.dot(self.user_embeddings[row[0]], self.item_embeddings[row[1]])
                temp = self.user_embeddings[row[0]]
                self.user_embeddings[row[0]] += self.lr * error * self.item_embeddings[row[1]]                          # 这儿是关键的训练迭代算法，但没咋看懂，甚至这儿还同时用item或user的嵌入向量交叉训练，去翻翻论文原文吧
                self.item_embeddings[row[1]] += self.lr * error * temp                                                  # 注意这里用来更新的user_embeddings要保证原汁原味哦

            error_counter += 1
            train_error = self.get_error(self.true_train_dataset)
            validation_error = self.get_error(self.validation_dataset)
            print('Training RMSE: {:.4f} Validation RMSE: {:.4f}'.format(train_error, validation_error))

            if validation_error < self.min_validation_error:
                self.min_validation_error = validation_error
                best_user_embeddings = copy.deepcopy(self.user_embeddings)                                              # 直接成群复制，python实在是太高效啦！
                best_item_embeddings = copy.deepcopy(self.item_embeddings)
                error_counter = 0
                best_epoch = _
            if error_counter >= early_stopping:                                                                         # 即已经过了局部最优点了，且规定次数之类也没能再找到另一个更小的局部最小点（但可能被一些非全局最小点的局部最小点吸收了，跑了几次结果都不一样，有遇到这种情况）
                break

        return best_user_embeddings,best_item_embeddings, best_epoch

    def initialize_latent_vectors(self):
        for row in self.true_train_dataset.iloc():                                                                      # (bug避坑)正确读取DataFrame中每行各元素的方法！啥 for user, item, rating in self.train_dataset 都是不可行的！！！
            self.user_embeddings.setdefault(row[0], self.generate_random_embeddings())                                  # 在下面一个类中执行，值为numpy数据类型
            self.item_embeddings.setdefault(row[1], self.generate_random_embeddings())                                  # 字典的setdefault函数，如果建不存在时（即user,item不存在时）则设置为默认值，否则返回已存在的值，因为在数据文件中同一个user可能会与多个item有联系，所以会在传入过程中多次出现该user值，但是一个user的初始嵌入向量只有一个嘛，item同理。

    def generate_random_embeddings(self):
        if self.initalization_method == 'random':
            return np.random.random_sample((self.n_latent,))
        elif self.initalization_method == 'normal':
            return np.random.normal(0, 0.1, (self.n_latent,))                                                           # 从“服从指定正态分布的序列”中随机取出指定个数的值。

    def get_error(self, dataset):
        total_error = 0
        counter = 0
        for row in dataset.iloc():
            if row[0] not in self.user_embeddings or row[1] not in self.item_embeddings:                                # 因为这儿分别计算得是train_data和test_data中的，所以data中的不一定都在
                continue
            total_error += (row[2] - np.dot(self.user_embeddings[row[0]], self.item_embeddings[row[1]])) ** 2
            counter += 1

        return np.sqrt(total_error / counter)