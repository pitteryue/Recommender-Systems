import scipy.spatial
import scipy.stats
import copy
import math
import pandas as pd
import numpy as np

class UBCF_implicit():

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

        for i in range(len(similarity_matrix.index)):
            U_knn = similarity_matrix.iloc[i,:].sort_values(ascending=False)[:self.K]
            U_knn_list = U_knn.index.tolist()
            U = U_knn.values.tolist()
            V = self.train_rating_matrix.loc[U_knn_list]
            estimated_rating_matrix.iloc[i,:] = np.dot(np.array(U), np.array(V))

        return estimated_rating_matrix

