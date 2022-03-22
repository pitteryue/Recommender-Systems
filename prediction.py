import numpy as np
import pandas as pd

class PREDICTION():

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def GET_ESTIMATED_RATING_MATRIX(self,user_embeddings,item_embeddings):
        self.user_embeddings_matrix = pd.DataFrame.from_dict(user_embeddings, orient='index')                           # user的embeddings矩阵是行矩阵，item的embeddings矩阵是列矩阵
        self.item_embeddings_matrix = pd.DataFrame.from_dict(item_embeddings)
        self.estimated_rating_matrix = self.user_embeddings_matrix.dot(self.item_embeddings_matrix)                     # 两个矩阵的索引在新矩阵中的行列上自动对上了
        # 处理train_dataset中出现过的数据
        for row in self.train_dataset.iloc():                                                                           # 鉴于从字典中找出训练集中出现过的数据并移除，耗时不说，语法也不完善，还不如就在此先把这些值自动排序时放到最后
            # 这里我处理了下，只要是出现在训练集的，都赋予一个很大的负值，从而等价于不参与排序，即不出现在推荐列表当中
            self.estimated_rating_matrix.loc[row['user'],row['item']] = -9999                                           # 注意！.loc()调用的是行列索引，但iloc()是行列号，从0开始的，不一样哦！
        return self.estimated_rating_matrix

    def GET_RECOMMENDATION_LIST(self, recommendation_size, all_recommendation_list):                                    # 字典没有直接的切片操作，所以要对每个键值下的list进行切片
        recommendation_list = {user: all_recommendation_list[user][: recommendation_size] for user in all_recommendation_list.keys()}

        return recommendation_list