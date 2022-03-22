import networkx as nx
import random
from networkx.algorithms import bipartite as bi
import numpy as np
from  BiNE_lsh import get_negs_by_lsh                                                                                   # 相当于载入lsh.py中的get_negs_by_lsh函数，from ... import ,,,也是从库中载入指定函数的方式
from io import open
import os
import itertools
import pandas as pd

import  BiNE_graph                                                                                                      # 这个是提前定义好的类

class GraphUtils(object):

    def __init__(self):
        # 创建空图，读取数据
        self.G = nx.Graph()                                                                                             # 创建空的简单
        self.edge_dict_u = {}                                                                                           # 将是个嵌套字典，用于记录用户u所关联的物品集及评分集
        self.edge_dict_v = {}                                                                                           # 将是个嵌套字典，用于记录物品v所关联的用户集及评分集
        self.edge_list = []                                                                                             # 仅仅记录三元组(user, item, rating)，而没有(item, user, rating)
        self.node_u = []
        self.node_v = []

        self.authority_u, self.authority_v = {}, {}
        self.walks_u, self.walks_v = [], []
        self.G_u, self.G_v = None, None
        self.fw_u = "homogeneous_u.dat"
        self.fw_v = "homogeneous_v.dat"
        self.negs_u = {}
        self.negs_v = {}
        self.context_u = {}
        self.context_v = {}

    # 这套代码中，构建图都喜欢用嵌套字典奥，以同时存储user, item, rating之间的信息~可以比较下我之前用的DataFrame，以后可以换一换
    # 嵌套dict的输出大概就是像这样的{user1:{item1, rating1},user2:{item2: rating2, item3: rating 3}, user3:{item4: rating 4}}
    def construct_training_graph(self, train_dataset):
        edge_list_u_v = []
        edge_list_v_u = []
        for row in train_dataset.iloc():
            user = row[0]
            item = row[1]
            rating = row[2]
            if self.edge_dict_u.get(user) is None:
                self.edge_dict_u[user] = {}
            if self.edge_dict_v.get(item) is None:
                self.edge_dict_v[item] = {}
            self.edge_dict_u[user][item] = float(rating)
            self.edge_dict_v[item][user] = float(rating)
            edge_list_u_v.append((user, item, float(rating)))
            edge_list_v_u.append((item, user, float(rating)))

        # create bipartite graph
        self.node_u = list(self.edge_dict_u.keys())                                                                     # 这里不能直接用字典进行排序！要先转换为list奥！
        self.node_v = list(self.edge_dict_v.keys())
        self.node_u.sort()                                                                                              # 这里为什么要进行排序呢？？？（为了后面在创建二部图A后的row_index以及col_index对应吧）
        self.node_v.sort()

        self.G.add_nodes_from(self.node_u, bipartite=0)
        self.G.add_nodes_from(self.node_v, bipartite=1)
        self.G.add_weighted_edges_from(edge_list_u_v + edge_list_v_u)                                                   # 加入权重需要用由元组构成的列表哦
        self.edge_list = edge_list_u_v

    def calculate_centrality(self, mode = 'hits'):
        if mode == 'degree_centrality':
            a = nx.degree_centrality(self.G)                                                                            # 统计图中节点的“度中心性”，输出a中记录了每个节点的指标数值，以后可用a[node]来检索
        else:
            h, a = nx.hits(self.G)                                                                                      # returns HITS hubs and authorities values for nodes

        # 下面的操作相当于对每个节点的中心性指标(authority)将进行归一化
        max_a_u, min_a_u, max_a_v, min_a_v = 0, 100000, 0, 100000
        for node in self.G.nodes():
            if node[0] == "u":                                                                                          # 因为数据集的user前有个u，item前有个i，比如user1对item1的rating为1，则表示为(u1, i1, 1)
                if max_a_u < a[node]:
                    max_a_u = a[node]
                if min_a_u > a[node]:
                    min_a_u = a[node]
            if node[0] == "i":
                if max_a_v < a[node]:
                    max_a_v = a[node]
                if min_a_v > a[node]:
                    min_a_v = a[node]
        for node in self.G.nodes():
            if node[0] == "u":
                if max_a_u - min_a_u != 0:
                    self.authority_u[node] = (float(a[node]) - min_a_u) / (max_a_u - min_a_u)                               # 这里是不是有问题？如果最小最大仍保留默认值，那么这个区间合理吗？？？
                else:
                    self.authority_u[node] = 0
            if node[0] == 'i':
                if max_a_v-min_a_v != 0:
                    self.authority_v[node] = (float(a[node]) - min_a_v) / (max_a_v - min_a_v)
                else:
                    self.authority_v[node] = 0

    def homogeneous_graph_random_walks(self, percentage, maxT, minT):                                                   # 输入参数分别是：“终止游走的概率”、“每个顶点出发的最大游走长度”和“每个顶点出发的最小游走长度”
        A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype = np.float, weight = 'weight', format = 'csr')# 将二部图转换为矩阵的形式，输出形式为：(0, 921)	1.0，存储的都是有连边的user和item，并且为1，即无权重
        row_index = dict(zip(self.node_u, itertools.count()))                                                           # 这两行相当于抽象建立了个DataFrame，即行号与索引的对应，和列号与列名的对应
        col_index = dict(zip(self.node_v, itertools.count()))
        index_row = dict(zip(row_index.values(), row_index.keys()))                                                     # 同样，记录这一行/列代表的是哪个用户/物品
        index_item = dict(zip(col_index.values(), col_index.keys()))

        # 下面这进行一次后直接调用输出的文件，因为太耗时了，不可能每次都跑一遍
        AT = A.transpose()
        # 下面用到了图论的一些知识，因为有边记为1无边记为0，所以重在连通性。下面做点积操作就能够得到在user/item同构图上的连通性了，PS：程序中主要耗时的地方就在这里，所以运行一次后就保存下来吧，后面直接调用文件即可。
        print("generating user homogeneous graph...")
        self.save_homogenous_graph_to_file(A.dot(AT), self.fw_u, index_row, index_row)                                  # 这里作为输出文件存储有啥用呢？（后面在分别在user/item所组成的同构图上随机游走时要使用）
        print("generating item homogeneous graph...")
        self.save_homogenous_graph_to_file(AT.dot(A), self.fw_v, index_item, index_item)


        # 下面的输入中，fw_u和fw_v是文件路径

        print("random walks on item homogeneous graph...")
        self.G_v, self.walks_v = self.get_random_walks_restart(self.fw_v, self.authority_v, percentage = percentage, maxT = maxT, minT = minT)
        print("random walks on user homogeneous graph...")
        self.G_u, self.walks_u = self.get_random_walks_restart(self.fw_u, self.authority_u, percentage = percentage, maxT=maxT, minT=minT)


    def save_homogenous_graph_to_file(self, A, datafile, index_row, index_item):
        (M, N) = A.shape
        csr_dict = A.__dict__                                                                                           # 类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的,,具体可以输出显示一下
        data = csr_dict.get("data")                                                                                     # 'data': array([1., 1., 1., ..., 1., 1., 1.])
        indptr = csr_dict.get("indptr")                                                                                 # 'indices': array([14985, 14983, 14961, ...,     7,     2,     1], dtype=int32),
        indices = csr_dict.get("indices")                                                                               # 'indptr': array([       0,     1404,     5402, ..., 20729601, 20732556, 20734178],dtype=int32)
        col_index = 0
        with open(datafile, 'w') as fw:
            for row in range(M):
                for col in range(indptr[row], indptr[row + 1]):
                    r = row
                    c = indices[col]
                    fw.write(index_row.get(r) + "\t" + index_item.get(c) + "\t" + str(data[col_index]) + "\n")          # 一顿操作就是在输出user/item的同构图，带有权重的，目前还不知道是怎么算出来的，但貌似是共同邻居的数量吧，前面貌似是为了加速把？先不用管太多
                    col_index += 1

    def get_random_walks_restart(self, datafile, hits_dict, percentage, maxT, minT):                                    # 这里的datafile是user/item的同构图哦
        G = BiNE_graph.load_edgelist(datafile, undirected = True)                                                            # 返回的是个嵌套字典，记录每一个node所关联的其他node，并且进行了去自环操作。
        print("number of nodes: {}".format(len(G.nodes())))                                                             # 并且这个G继承了graph_utils
        print("walking...")
        walks = BiNE_graph.build_deepwalk_corpus_random(G, hits_dict, percentage = percentage, maxT = maxT, minT = minT, alpha = 0)
                                                                                                                        # 这里的传入参数hits_dict就是之前得到的节点中心性指标list，里面都是在0到1之间的，walks是一个list
        print("walking...ok")
        return G, walks

    def get_negs(self,num_negs):
        self.negs_u, self.negs_v = get_negs_by_lsh(self.edge_dict_u, self.edge_dict_v, num_negs)                        # 前俩参数都是(user, item, rating)和(item, user, rating)嵌套dict
        return self.negs_u, self.negs_v

    def get_context_and_negatives(self, G, walks, win_size, num_negs, negs_dict):                                       # 这里的输入也就是在homogeneous graph上随机游走后的结果
        if isinstance(G, BiNE_graph.Graph):                                                                             # 测试对象 G 是否属于指定类型 graph.Graph
            node_list = list(G.nodes())
        elif isinstance(G, list):
            node_list = list(G)

        word2id = {}
        for i in range(len(node_list)):
            word2id[node_list[i]] = i + 1                                                                               # 输出形如：{'i10': 1, 'i46': 2, 'i99': 3, ...}

        walk_list = walks
        print("context...")
        context_dict = {}                                                                                               # 一个node，它会在不同的walks中有不一样的context，因此建立一个全局字典，将它们联系在一起
        new_neg_dict = {}
        for step in range(len(walk_list)):
            walk = walk_list[step % len(walk_list)]                                                                     # 里面的step % len(walk_list)算出的数值其实就是step，从0开始的奥
            for iter in range(len(walk)):                                                                               # iter可视为走到walk中每个node位置的光标。其目的是对于每一条walk，以其中的每个node为中心，wind_size内的nodes为context，创造样本。一定是对于每一个node奥
                start = max(0, iter - win_size)                                                                         # 要保证不能超出下界
                end = min(len(walk), iter + win_size + 1)                                                               # 要保证不能超出上界
                if context_dict.get(walk[iter]) is None:
                    context_dict[walk[iter]] = []                                                                       # 相当于一个node，它会在不同的walks中有不一样的context，因此建立一个全局字典，将它们联系在一起
                    new_neg_dict[walk[iter]] = []
                labels_list = []                                                                                        # 这下面开始没咋看懂，不过可以先不管，知道最后返回输出的是俩嵌套字典就行
                negs = negs_dict[walk[iter]]
                for index in range(start, end):
                    if walk[index] in negs:
                        negs.remove(walk[index])
                    if walk[index] == walk[iter]:
                        continue
                    else:
                        labels_list.append(walk[index])
                neg_sample = random.sample(negs,min(num_negs,len(negs)))
                context_dict[walk[iter]].append(labels_list)
                new_neg_dict[walk[iter]].append(neg_sample)
        print("context...ok")
        return context_dict, new_neg_dict
