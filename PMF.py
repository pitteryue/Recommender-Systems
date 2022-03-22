import numpy as np
from numpy.random import RandomState
import pandas as pd
import copy
import pickle as pkl
from numpy.linalg import norm

class PMF():

    def __init__(self, true_train_dataset, validation_dataset, user_embeddings, item_embeddings, n_latent, max_epochs, lr, momuntum = 0, initialization_method = 'normal',mean = 0, std = 0.1):
        self.true_train_dataset = true_train_dataset
        self.validation_dataset = validation_dataset
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.max_epochs = max_epochs
        self.n_latent = n_latent
        self.lr = lr
        self.momuntum = momuntum                                                                                        # momuntum是动量因子
        self.initialization_method = initialization_method
        self.mean = mean
        self.std = std

    def train(self, early_stopping, lambda_alpha = 0.01, lambda_beta = 0.01):                                           # 这儿泛化用的是正则化方法
        self.initialize_latent_vectors()                                                                                # 到这就可以打印初始嵌入向量了，即 print(self.user_features,self.item_features)
        self.min_validation_error = np.inf
        error_counter = 0
        best_epoch = 0
        # 初始化动量因子
        momuntum_u = np.zeros(self.n_latent)
        momuntum_v = np.zeros(self.n_latent)

        for _ in range(self.max_epochs):
            for row in self.true_train_dataset.iloc():
                grads_u = np.dot((row['rating'] - np.dot(self.user_embeddings[row['user']], self.item_embeddings[row['item']])),- self.item_embeddings[row['item']]) + lambda_alpha * self.user_embeddings[row['user']]
                grads_v = np.dot((row['rating'] - np.dot(self.user_embeddings[row['user']], self.item_embeddings[row['item']])), - self.user_embeddings[row['user']]) + lambda_beta * self.item_embeddings[row['item']]
                momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
                momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
                self.user_embeddings[row['user']] = self.user_embeddings[row['user']] - momuntum_u
                self.item_embeddings[row['item']] = self.item_embeddings[row['item']] - momuntum_v

            error_counter += 1
            train_error = self.get_error(self.true_train_dataset)
            validation_error = self.get_error(self.validation_dataset)
            #print('Training RMSE: {:.4f} Validation RMSE: {:.4f}'.format(train_error, validation_error))

            if validation_error < self.min_validation_error:
                self.min_validation_error = train_error
                best_user_embeddings = copy.deepcopy(self.user_embeddings)                                              # 直接成群复制，python实在是太高效啦！
                best_item_embeddings = copy.deepcopy(self.item_embeddings)
                error_counter = 0
                best_epoch = _
            if error_counter >= early_stopping:                                                                         # 即已经过了局部最小点了，且规定次数之类也没能再找到另一个更小的局部最小点（但可能被一些非全局最小点的局部最小点吸收了，跑了几次结果都不一样，有遇到这种情况）
                break

        return best_user_embeddings,best_item_embeddings, best_epoch

    def initialize_latent_vectors(self):
        for row in self.true_train_dataset.iloc():
            self.user_embeddings.setdefault(row[0], self.generate_random_embeddings())
            self.item_embeddings.setdefault(row[1], self.generate_random_embeddings())

    def generate_random_embeddings(self):
        # Generate features depending on given method
        if self.initialization_method == 'random':
            return np.random.random_sample((self.n_latent,))
        elif self.initialization_method == 'normal':
            return np.random.normal(self.mean, self.std, (self.n_latent,))

    def get_error(self, data):
        total_error = 0
        counter = 0
        for row in data.iloc():
            if row[0] not in self.user_embeddings or row[1] not in self.item_embeddings:
                continue
            total_error += (row[2] - np.dot(self.user_embeddings[row[0]], self.item_embeddings[row[1]])) ** 2
            counter += 1

        return np.sqrt(total_error / counter)