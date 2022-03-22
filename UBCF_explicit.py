import scipy.spatial
import scipy.stats
import copy
import math
import pandas as pd
import numpy as np

class UBCF_explicit():

    def __init__(self, similarity_metric_name, train_rating_matrix, K):
        self.similarity_metric_name = similarity_metric_name
        self.K = K
        self.train_rating_matrix = train_rating_matrix

    def similarity_matrix(self):
        if self.similarity_metric_name == "pearsonr":
            similarity_matrix_np = np.corrcoef(np.array(self.train_rating_matrix))
            similarity_matrix = pd.DataFrame(similarity_matrix_np, index=self.train_rating_matrix.index.values, columns=self.train_rating_matrix.index.values)

        return similarity_matrix

    def prediction(self, similarity_matrix):
        for i in range(len(similarity_matrix.index)):                                                                   # 为了避免在k近邻中加入自己
            similarity_matrix.iloc[i,i] = -9999
        estimated_rating_matrix = pd.DataFrame(0, index = self.train_rating_matrix.index.values, columns = self.train_rating_matrix.columns.values)

        mean_rating = []
        for i in range(len(self.train_rating_matrix.index)):
            total = 0
            count = 0
            for j in range(len(self.train_rating_matrix.columns)):
                if self.train_rating_matrix.iloc[i,j]!=0:
                    total += self.train_rating_matrix.iloc[i,j]
                    count += 1

            if count == 0:                                                                                              # 注意这里要排除分母为0的情况
                mean = 0
            else:
                mean = total / count

            mean_rating.append(mean)

        for i in range(len(self.train_rating_matrix.index)):                                                            # 注意！是对于非零元素而言的！
            for j in range(len(self.train_rating_matrix.columns)):
                if self.train_rating_matrix.iloc[i,j]!=0:
                    self.train_rating_matrix.iloc[i,j] -= mean_rating[i]                                                # 注意！对于0元素，即没有打分记录的元素，不用减！而且该操作必须放到算完相似矩阵后再进行！！！

        for i in range(len(similarity_matrix.index)):
            U_knn = similarity_matrix.iloc[i,:].sort_values(ascending=False)[:self.K]
            U_knn_list = U_knn.index.tolist()
            U = U_knn.values.tolist()
            V = self.train_rating_matrix.loc[U_knn_list]
            abs_U = map(abs,U)                                                                                          # 对list中每一项要取绝对值的哦！
            sum_abs_U = sum(abs_U)
            k = 1.0 / sum_abs_U                                                                                         # 注意分母一定不能是表达式，否则会报错的！
            estimated_rating_matrix.iloc[i,:] = mean_rating[i] + k * np.dot(np.array(U), np.array(V))

        return estimated_rating_matrix