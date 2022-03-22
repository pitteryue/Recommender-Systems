import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split                                                                    # 随机划分训练集和测试集的时候用，一行代码就搞定了
import csv
import networkx as nx
import matplotlib.pyplot as plt

class DATA_RATINGS():

    def __init__(self,test_portion):
        self.test_portion=test_portion

    def TRAIN_TEST_BY_RANDOM(self,input_csv_path,output_csv_train_path,output_csv_test_path):
        dataset = pd.read_table(input_csv_path, sep=',',engine='python')
        self.train_dataset,self.test_dataset = train_test_split(dataset, test_size=self.test_portion)

        '''
        # 如果要按照时间戳来抽取训练集和测试集，那么直接进行切片操作即可。注意：前提是将输入数据按时间戳升序排好！
        train_dataset = dataset[:math.ceil(dataset.shape[0] * self.test_portion - 1)]                                   
        test_dataset = dataset[math.ceil(dataset.shape[0] * self.test_portion):dataset.shape[0]]
        '''
        self.train_dataset.columns=['user','item','rating']                                                             # 修改列名
        self.test_dataset.columns=['user','item','rating']
        self.train_dataset.to_csv(output_csv_train_path, index=False)
        self.test_dataset.to_csv(output_csv_test_path, index=False)

        return self.train_dataset,self.test_dataset

    def TRAIN_VALIDATION_TEST_BY_TIME(self, ratings_dataset, train_portion = 0.8, validation_portion = 0.1, test_portion = 0.1):
        size = len(ratings_dataset.index)
        train_size = math.floor(train_portion * size)
        validation_size = math.floor((train_portion + validation_portion) * size)
        test_size = math.floor((train_portion + validation_portion + test_portion) * size)

        train_set = ratings_dataset.iloc[0 : train_size]
        validation_set = ratings_dataset.iloc[train_size + 1 : validation_size]
        test_set = ratings_dataset.iloc[validation_size + 1 : test_size]

        train_set.to_csv('train_ratings.csv', index = False)
        validation_set.to_csv('validation_ratings.csv', index = False)
        test_set.to_csv('test_ratings.csv', index = False)

    def TRAIN_VALIDATION_TEST_BY_TIME_AND_LAYER(self, ratings_dataset, train_portion = 0.8, validation_portion = 0.1, test_portion = 0.1):
        total = {}
        for row in ratings_dataset.iloc():
            user = int(row['user'])
            item = int(row['item'])
            rating = float(row['rating'])
            if total.get(user) is None:
                total[user] = {}
            total[user][item] = rating

        with open('train_ratings.csv', 'w', newline = '') as train_set:
            with open('validation_ratings.csv', 'w', newline='') as validation_set:
                with open('test_ratings.csv', 'w', newline='') as test_set:
                    train_writer = csv.writer(train_set,delimiter=',')
                    validation_writer = csv.writer(validation_set,delimiter=',')
                    test_writer = csv.writer(test_set,delimiter=',')
                    for user, item_rating in total.items():                                                                               # 注意这里的values也是一个字典奥！它的len就是其中包含的key，也就是items的个数
                        size = len(item_rating)
                        train_size = math.floor(train_portion * size)  # 下取整也要在math里来调用哦！
                        validation_size = math.floor((train_portion + validation_portion) * size)
                        test_size = math.floor((train_portion + validation_portion + test_portion) * size)

                        count = 0
                        for item, rating in item_rating.items():
                            if count <= train_size:
                                train_writer.writerow([user, item, rating])
                            elif count > train_size and count <= validation_size:
                                validation_writer.writerow([user, item, rating])
                            elif count > validation_size and count <= test_size:
                                test_writer.writerow([user, item, rating])
                            count += 1

    # 将.dat类型数据集转换为.csv类型数据集
    def DAT_TO_CSV(self,input_dat_path,outout_csv_path):                                                                # 其实要读取.dat文件，也不用先转换成.csv，只要设置sep='::'就可以了！
        dataset = pd.read_table(input_dat_path, sep='\t', header=None,engine='python')                                   # read_table()返回一个DataFrame，其中sep标识读取文件中的分隔符，header=None表示读取文件的第一行不是列的名字。注意：最后加上engine='python'，否则会有bug
        dataset.to_csv(outout_csv_path, index=False)                                                                    # index=True则在输出文件中会有行索引和列名(都默认为从0开始的有序数)

    # 为训练集构建rating matrix，且不保证user和item的编号是连续的。其实这里还能引申！比如只保留与至少五个items交互过的用户等等可以通过删去不满足条件的行或列来实现
    def GET_TRAIN_RATING_MATRIX(self, train_dataset):
        max_user = max(train_dataset['user'])
        max_item = max(train_dataset['item'])
        train_dataset_rating_matrix = pd.DataFrame(np.zeros((max_user+1,max_item+1)))                                   # 生成全0的dataframe矩阵，注意！数据中是从1开始编号的
        for row in train_dataset.iloc():
            train_dataset_rating_matrix.loc[row['user'],row['item']] = row['rating']
        train_dataset_rating_matrix = train_dataset_rating_matrix.loc[~(train_dataset_rating_matrix==0).all(axis = 1)]  # 删除全0行，这里非常巧妙，用.sum()来判断的！或者是df=df[df.values.sum(axis=1)!=0]
        train_dataset_rating_matrix = train_dataset_rating_matrix.loc[:, (train_dataset_rating_matrix != 0).any(axis=0)]# 删除全0列

        return train_dataset_rating_matrix

    def ONE_HOT_ENCODING(self,train_dataset):
        max_user = max(train_dataset['user'])
        max_item = max(train_dataset['item'])
        user_one_hot = pd.DataFrame(np.zeros((max_user + 1, max_user + 1)))
        item_one_hot = pd.DataFrame(np.zeros((max_item + 1, max_item + 1)))
        for row in train_dataset.iloc():
            user_one_hot.loc[row['user']][row['user']] = 1
            item_one_hot.loc[row['item']][row['item']] = 1
        user_one_hot = user_one_hot.loc[~(user_one_hot==0).all(axis=1)]                                                 # 剔除不存在的id
        user_one_hot = user_one_hot.loc[:,(user_one_hot != 0).any(axis=0)]
        item_one_hot = item_one_hot.loc[~(item_one_hot == 0).all(axis=1)]
        item_one_hot = item_one_hot.loc[:, (item_one_hot != 0).any(axis=0)]

        return user_one_hot,item_one_hot

    def OBTAIN_NEGATIVE_SAMPLES(self, train_dataset, train_rating_matrix):                                              # 为训练样本中每条记录所对应的用户来以此建立负样本
        negative_samples = pd.DataFrame()                                                                               # 这里不能指定列名，因为生成的数据不是字典哦！和列名对不上，否则就成了6列了
        user_set = pd.Series(train_rating_matrix.index.values)                                                          # 要先转换成dataframe/series才好调用sample哦
        item_set = pd.Series(train_rating_matrix.columns.values)

        for i in range(len(train_dataset.index)):                                                                       # 为每个用户来构建负样本
            user = train_dataset.iloc[i,0]
            while True:                                                                                                 # 死循环（或者说是条件暂停循环）的写法
                item = int(item_set.sample(1).values)
                if train_rating_matrix.loc[user, item] == 0:
                    negative_samples = negative_samples.append(pd.Series([user, item, 0]), ignore_index=True)
                    break

        negative_samples.columns = ['user','item', 'rating']                                                            # 这里要改一下列名，否则默认的0,1,2
        negative_samples.to_csv("negative_samples.csv", index = False)

        return negative_samples

    def PLOT_DEGREE_DISTRIBUTION(self, ratings):
        G = nx.Graph()

        ratings_u_v_dict = {}
        ratings_v_u_dict = {}
        edge_list_u_v = []
        edge_list_v_u = []
        for row in ratings.iloc():
            user = row['user']
            item = row['item']
            rating = row['rating']
            if ratings_u_v_dict.get(user) is None:
                ratings_u_v_dict[user] = {}
            if ratings_v_u_dict.get(item) is None:
                ratings_v_u_dict[item] = {}
            ratings_u_v_dict[user][item] = rating
            ratings_v_u_dict[item][user] = rating

            edge_list_u_v.append((user, item, rating))
            edge_list_v_u.append((item, user, rating))

        node_u = list(ratings_u_v_dict.keys())
        node_v = list(ratings_v_u_dict.keys())
        node_u.sort()
        node_v.sort()
        G.add_nodes_from(node_u, bipartite = 0)
        G.add_nodes_from(node_v, bipartite = 1)
        G.add_weighted_edges_from(edge_list_u_v + edge_list_v_u)                                                        # 并不会影响度分布，还是当做的相同的一条单边

        degree = nx.degree_histogram(G)
        x = range(len(degree))                                                                                          # 生成x轴序列，从1到最大度
        #y = [z / float(sum(degree)) for z in degree]                                                                    # 将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
        plt.loglog(x, degree, color="blue", linewidth=1)                                                                     # 在双对数坐标轴上绘制度分布曲线
        plt.show()                                                                                                      # 显示图表

    def RENUMBER_USER(self,dataset):

        # 输出数据集中的原本情况
        max_users = max(dataset['user'])
        max_items = max(dataset['item'])                                                                                # 这里的前提是原数据集的user和item都是从1开始编号的，如果是0，则需要额外再+1！！！
        valid_users = set(dataset.iloc[:, 0].tolist())
        valid_items = set(dataset.iloc[:, 1].tolist())
        valid_users = len(valid_users)
        valid_items = len(valid_items)
        print('before renumber:' )
        print('max users: ', max_users, 'max_items: ', max_items)
        print('valid_users', valid_users, 'valid_items', valid_items)

        user_posi = 1
        last_user = dataset.iloc[0,0]
        for i in range(len(dataset.index)):
            # 对user进行重编号操作
            if dataset.iloc[i,0] != last_user:
                last_user = dataset.iloc[i,0]
                user_posi+= 1
                dataset.iloc[i,0] = user_posi
            elif dataset.iloc[i,0] == last_user:
                dataset.iloc[i, 0] = user_posi

        # 输出数据集重编号后的情况
        max_users = max(dataset['user'])
        max_items = max(dataset['item'])
        valid_users = set(dataset.iloc[:, 0].tolist())
        valid_items = set(dataset.iloc[:, 1].tolist())
        valid_users = len(valid_users)
        valid_items = len(valid_items)
        print('after renumber:')
        print('max users: ', max_users, 'max_items: ', max_items)
        print('valid_users', valid_users, 'valid_items', valid_items)

        dataset.to_csv('ratings_user_renumber.csv', sep = ',', index = False)

    def RENUMBER_ITEM(self, dataset):

        # 输出数据集中的原本情况
        max_users = max(dataset['user'])
        max_items = max(dataset['item'])  # 这里的前提是原数据集的user和item都是从1开始编号的，如果是0，则需要额外再+1！！！
        valid_users = set(dataset.iloc[:, 0].tolist())
        valid_items = set(dataset.iloc[:, 1].tolist())
        valid_users = len(valid_users)
        valid_items = len(valid_items)
        print('before renumber:')
        print('max users: ', max_users, 'max_items: ', max_items)
        print('valid_users', valid_users, 'valid_items', valid_items)

        item_posi = 1
        last_item = dataset.iloc[0, 1]
        for i in range(len(dataset.index)):
            # 对item进行重编号操作
            if dataset.iloc[i, 1] != last_item:
                last_item = dataset.iloc[i, 1]
                item_posi += 1
                dataset.iloc[i, 1] = item_posi
            elif dataset.iloc[i, 1] == last_item:
                dataset.iloc[i, 1] = item_posi

        # 输出数据集重编号后的情况
        max_users = max(dataset['user'])
        max_items = max(dataset['item'])
        valid_users = set(dataset.iloc[:, 0].tolist())
        valid_items = set(dataset.iloc[:, 1].tolist())
        valid_users = len(valid_users)
        valid_items = len(valid_items)
        print('after renumber:')
        print('max users: ', max_users, 'max_items: ', max_items)
        print('valid_users', valid_users, 'valid_items', valid_items)

        dataset.to_csv('ratings_user_and_item_renumber.csv', sep=',', index = False)
