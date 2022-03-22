import numpy as np
import pandas as pd
import copy

class PhysRS():

    def __init__(self, train_rating_matrix):
        self.train_rating_matrix = train_rating_matrix
        self.train_rating_matrix_T = train_rating_matrix.transpose()

    def HC_propogation(self):
        user_degree_train_rating_matrix_T = copy.deepcopy(self.train_rating_matrix_T)
        for i in range(len(user_degree_train_rating_matrix_T.columns)):                                                 # 注意下面不要用self.train_rating_matrix_T[i].要用.iloc[:,i]，否则要报错！因为[i]中的i是列名！不是列号！！！
            if 1 in user_degree_train_rating_matrix_T.iloc[:, i].value_counts():
                user_degree = user_degree_train_rating_matrix_T.iloc[:,i].value_counts().loc[1]                         # 这里对pd统计后得到的是一个Series，然后可通过.loc[]获取1的出现次数统计结果
                user_degree_train_rating_matrix_T.iloc[:, i] = user_degree_train_rating_matrix_T.iloc[:, i] / user_degree
            else:
                user_degree_train_rating_matrix_T.iloc[:, i] = 0

        item_degree_train_rating_matrix = copy.deepcopy(self.train_rating_matrix)
        for j in range(len(item_degree_train_rating_matrix.columns)):
            if 1 in item_degree_train_rating_matrix.iloc[:,j].value_counts():
                item_degree = item_degree_train_rating_matrix.iloc[:,j].value_counts().loc[1]                           # 还要考虑到这一列没有1的情况，即在value_counts输出的Series中没有1这个索引！
                item_degree_train_rating_matrix.iloc[:, j] = item_degree_train_rating_matrix.iloc[:, j] / item_degree
            else:
                item_degree_train_rating_matrix.iloc[:, j] = 0

        estimated_rating_matrix_np = np.dot(np.dot(np.array(self.train_rating_matrix), np.array(user_degree_train_rating_matrix_T)), item_degree_train_rating_matrix)
        estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_np, index = self.train_rating_matrix.index.values, columns = self.train_rating_matrix.columns.values)

        return estimated_rating_matrix

    def MD_propogation(self):
        item_degree_train_rating_matrix = copy.deepcopy(self.train_rating_matrix)
        for i in range(len(item_degree_train_rating_matrix.columns)):
            if 1 in item_degree_train_rating_matrix.iloc[:,i].value_counts():
                item_degree = item_degree_train_rating_matrix.iloc[:,i].value_counts().loc[1]
                item_degree_train_rating_matrix.iloc[:,i] = item_degree_train_rating_matrix.iloc[:,i] / item_degree
            else:
                item_degree_train_rating_matrix.iloc[:,i] = 0

        user_degree_train_rating_matrix = copy.deepcopy(self.train_rating_matrix_T)
        for j in range(len(user_degree_train_rating_matrix.columns)):
            if 1 in user_degree_train_rating_matrix.iloc[:,j].value_counts():
                user_degree = user_degree_train_rating_matrix.iloc[:,j].value_counts().loc[1]
                user_degree_train_rating_matrix.iloc[:,j] = user_degree_train_rating_matrix.iloc[:,j] / user_degree
        user_degree_train_rating_matrix = user_degree_train_rating_matrix.transpose()

        estimated_rating_matrix_np = np.dot(np.dot(np.array(item_degree_train_rating_matrix), np.array(self.train_rating_matrix_T)),user_degree_train_rating_matrix)
        estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_np, index=self.train_rating_matrix.index.values, columns=self.train_rating_matrix.columns.values)

        return estimated_rating_matrix

    # 相当于将MD的第一步（即对item的1个资源除以item的度值）放到HC之前
    def HC_MD_propogation(self, para_lambda):
        new_train_rating_matrix = copy.deepcopy(self.train_rating_matrix)
        for i in range(len(new_train_rating_matrix.columns)):
            if 1 in new_train_rating_matrix.iloc[:, i].value_counts():
                item_degree = new_train_rating_matrix.iloc[:, i].value_counts().loc[1] ** para_lambda
                new_train_rating_matrix.iloc[:, i] = new_train_rating_matrix.iloc[:, i] / item_degree
            else:
                new_train_rating_matrix.iloc[:, i] = 0

        user_degree_train_rating_matrix_T = copy.deepcopy(self.train_rating_matrix_T)
        for i in range(len(user_degree_train_rating_matrix_T.columns)):
            if 1 in user_degree_train_rating_matrix_T.iloc[:, i].value_counts():
                user_degree = user_degree_train_rating_matrix_T.iloc[:, i].value_counts().loc[1]
                user_degree_train_rating_matrix_T.iloc[:, i] = user_degree_train_rating_matrix_T.iloc[:,i] / user_degree
            else:
                user_degree_train_rating_matrix_T.iloc[:, i] = 0

        item_degree_train_rating_matrix = copy.deepcopy(self.train_rating_matrix)
        for j in range(len(item_degree_train_rating_matrix.columns)):
            if 1 in item_degree_train_rating_matrix.iloc[:, j].value_counts():
                item_degree = item_degree_train_rating_matrix.iloc[:, j].value_counts().loc[1] ** (1 - para_lambda)
                item_degree_train_rating_matrix.iloc[:, j] = item_degree_train_rating_matrix.iloc[:, j] / item_degree
            else:
                item_degree_train_rating_matrix.iloc[:, j] = 0

        estimated_rating_matrix_np = np.dot(np.dot(np.array(new_train_rating_matrix), np.array(user_degree_train_rating_matrix_T)),item_degree_train_rating_matrix)
        estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_np, index=self.train_rating_matrix.index.values,columns=self.train_rating_matrix.columns.values)

        return estimated_rating_matrix