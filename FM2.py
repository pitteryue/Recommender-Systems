import numpy as np
import pandas as pd
import copy
import pickle as pkl
from numpy.linalg import norm

class FM2():

    def __init__(self, true_train_dataset , validation_dataset, user_embeddings, item_embeddings,  n_latent, max_epochs, lr):
        self.true_train_dataset = true_train_dataset
        self.validation_dataset = validation_dataset

        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.lr = lr

        self.max_user = max(true_train_dataset['user'])
        self.max_item = max(true_train_dataset['item'])

        self.w0 = 0                                                                                                     # 这5个都是待训练的parameters及其初始化
        self.w_user = np.zeros((self.max_user + 1, 1))
        self.w_item = np.zeros((self.max_item + 1, 1))
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.initialize_latent_vectors()

    def train(self, early_stopping):
        self.min_validation_error = np.inf
        error_counter = 0
        best_epoch = 0
        for _ in range(self.max_epochs):
            for row in self.true_train_dataset.iloc():
                interaction_1 = self.user_embeddings[row['user']] + self.item_embeddings[row['item']]
                interaction_2 = np.multiply(self.user_embeddings[row['user']], self.user_embeddings[row['user']]) + \
                                np.multiply(self.item_embeddings[row['item']], self.item_embeddings[row['item']])       # np.multiply()函数是对矩阵或向量对应位置相乘，输出的大小与输入一致
                interaction = np.sum(np.multiply(interaction_1, interaction_1) - interaction_2)/2
                y = self.w0 + (self.w_user[int(row['user'])] + self.w_item[int(row['item'])]) + interaction             # 模型的预测值

                error = row['rating'] - y                                                                               # 在这里我是用的RMSE+SGD，但原始代码用的损失函数和优化方法都有疑惑，不知怎么构造和推导的？？？
                self.w0 = self.w0 + self.lr * error
                self.w_user[int(row['user'])] = self.w_user[int(row['user'])] + self.lr * error
                self.w_item[int(row['item'])] = self.w_item[int(row['item'])] + self.lr * error
                tmp = self.user_embeddings[row['user']]
                self.user_embeddings[row['user']] = self.user_embeddings[row['user']] + self.lr * error * self.item_embeddings[row['item']]
                self.item_embeddings[row['item']] = self.item_embeddings[row['item']] + self.lr * error * tmp

            error_counter += 1
            train_error = self.get_error(self.true_train_dataset)
            validation_error = self.get_error(self.validation_dataset)
            #print('Training RMSE:', train_error, 'Validation RMSE:', validation_error)

            if validation_error < self.min_validation_error:
                self.min_validation_error = validation_error
                best_w0 = copy.deepcopy(self.w0)
                best_w_user = copy.deepcopy(self.w_user)
                best_w_item = copy.deepcopy(self.w_item)
                best_user_embeddings = copy.deepcopy(self.user_embeddings)
                best_item_embeddings = copy.deepcopy(self.item_embeddings)
                error_counter = 0
                best_epoch = _

            if error_counter >= early_stopping:
                break

        return best_w0, best_w_user, best_w_item, best_user_embeddings, best_item_embeddings, best_epoch

    def initialize_latent_vectors(self):
        for row in self.true_train_dataset.iloc():
            self.user_embeddings.setdefault(row['user'], self.generate_random_embeddings())
            self.item_embeddings.setdefault(row['item'], self.generate_random_embeddings())

    def generate_random_embeddings(self):
        return np.random.normal(0, 0.2, (self.n_latent,))

    def get_error(self, dataset):
        total_error = 0
        counter = 0
        for row in dataset.iloc():
            if row[0] not in self.user_embeddings or row[1] not in self.item_embeddings:                                # 因为这儿分别计算得是train_data和test_data中的，所以data中的不一定都在
                continue

            interaction_1 = self.user_embeddings[row['user']] + self.item_embeddings[row['item']]
            interaction_2 = np.multiply(self.user_embeddings[row['user']], self.user_embeddings[row['user']]) + \
                            np.multiply(self.item_embeddings[row['item']], self.item_embeddings[row['item']])
            interaction = np.sum(np.multiply(interaction_1, interaction_1) - interaction_2) / 2
            predicted_y = self.w0 + (self.w_user[int(row['user'])] + self.w_item[int(row['item'])]) + interaction

            total_error += (row[2] - predicted_y) ** 2
            counter += 1

        return np.sqrt(total_error / counter)