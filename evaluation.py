import numpy as np
import pandas as pd
import math

class EVALUATION():

    def __init__(self,estimated_rating_matrix, recommmendation_size, recommendation_list, train_dataset, valid_items, test_dataset):
        self.recommendation_size = recommmendation_size
        self.estimated_rating_matrix = estimated_rating_matrix
        self.recommendation_list = recommendation_list
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.valid_items = valid_items

        self.user_size = len(self.recommendation_list.keys())

    def MERICS_01(self):
        self.avi_test_dataset_size = 0
        self.total_hits = 0
        self.total_NDCG = 0
        self.total_abs_deviation_of_ratings = 0
        self.total_square_deviation_of_ratings = 0

        for row in self.test_dataset.iloc():
            if row[0] in self.recommendation_list.keys() and row[1] in self.valid_items:                                # 修复BUG：别忘了item也必须要在训练集中出现过才行哦！
                self.avi_test_dataset_size += 1
                if row[1] in self.recommendation_list[row[0]].values:
                    self.total_hits += 1
                    list_for_this_user = self.recommendation_list[row[0]].values.tolist()
                    posi = list_for_this_user.index(row[1])
                    self.total_NDCG += np.reciprocal(np.log2(posi + 2))                                                 # np.reciprocal()函数返回参数逐元素的倒数（这里+2是因为索引位置从0开始的吧，否则会出现分母=log2(1)=0的情况）

                # 修复BUG：下面这里还要算上没有命中的那一部分！如果只算命中的，当然误差很小啦！！
                self.total_abs_deviation_of_ratings += abs(self.estimated_rating_matrix.loc[row[0],row[1]] - row[2])
                self.total_square_deviation_of_ratings += (self.estimated_rating_matrix.loc[row[0],row[1]] - row[2])**2

        self.Precision = self.total_hits / (self.recommendation_size * self.user_size)
        self.HR = self.total_hits / self.avi_test_dataset_size
        self.MAE = self.total_abs_deviation_of_ratings / self.avi_test_dataset_size
        self.RMSE = math.sqrt(self.total_square_deviation_of_ratings / self.avi_test_dataset_size)
        self.NDCG = self.total_NDCG / self.avi_test_dataset_size

        return self.Precision, self.HR, self.NDCG, self.MAE, self.RMSE

    # 下面是METRICS_02，因为按照我的评分矩阵的布局，只要是出现在训练集中的，都给了个无穷大的负值，所以FP一定等于0，这是不是一般都不用precision, recall, F1那些指标的原因呢？
    '''
    def METRICS_02(self):
        # 下面的数据都是将在Metrics中共用的信息
        self.size_of_train_dataset = self.train_dataset.shape[0]
        self.size_of_test_dataset = self.test_dataset.shape[0]                                                          # 注意shape后面是[]而不是()
        self.size_of_recommendation_list = len(self.recommendation_list) * self.recommendation_size                     # len(dict)可用于返回字典"键"的个数
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.abs_deviation_of_ratings = 0
        self.square_deviation_of_ratings = 0

        for row in self.test_dataset.iloc():
            if row['user'] in self.recommendation_list.keys():
                if row['item'] in self.recommendation_list[row['user']].values:                                         # dict.items()输出的是整个字典，所以这里要这样判断
                    self.TP = self.TP + 1
                    self.abs_deviation_of_ratings = self.abs_deviation_of_ratings + abs(self.estimated_rating_matrix.loc[row['user'], row['item']] - row['rating'])
                    self.square_deviation_of_ratings = self.square_deviation_of_ratings + (self.estimated_rating_matrix.loc[row['user'],row['item']] - row['rating'])**2
                else:
                    self.TN = self.TN + 1

        for row in self.train_dataset.iloc():
            if row['user'] in self.recommendation_list.keys():
                if row['item'] in self.recommendation_list[row['user']].values:
                    self.FP = self.FP + 1
                else:
                    self.FN = self.FN + 1


        # print('TP=',self.TP, 'TN=', self.TN, 'FP=', self.FP, 'FN=', self.FN)
        # print('list_size=',self.size_of_recommendation_list,'TP+FP=', self.TP + self.FP)
        # print('not_list_size=', self.size_of_train_dataset - self.size_of_recommendation_list, 'TN+FN=', self.TN + self.FN)
        # print('train_dataset_size=', self.size_of_train_dataset, 'FP+FN=', self.FP + self.FN)
        # print('test_dataset_size=', self.size_of_test_dataset, 'TP+TN=',self.TP + self.TN)

        # 以下是指标的计算
        self.precision = self.TP / (self.TP + self.FP)
        self.recall = self.TP / (self.TP + self.FN)
        self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)
        self.MAE =  (1 / self.size_of_test_dataset) * self.abs_deviation_of_ratings                                     # 这里有BUG，应该用参与比较的那些test来作为分母！
        self.RMAE = math.sqrt((1 / self.size_of_test_dataset)* self.square_deviation_of_ratings)

        return self.precision, self.recall, self.F1, self.MAE, self.RMAE
    '''