from datasketch import MinHashLSHForest, MinHash, MinHashLSH                                                            # datasketch gives you probabilistic data structures that can process and search very large amount of data super fast, with little loss of accuracy.所以说是一种近似算法
import random

def get_negs_by_lsh(user_dict, item_dict, num_negs):                                                                    # 参数即 self.edge_dict_u 和 self.edge_dict_v，即为嵌套字典
    sample_num_u = max(300, int(len(user_dict) * 0.01 * num_negs))                                                      # 这个负样本数的长度为什么这么设定呢？（其中len是统计字典中key的种数）
    sample_num_v = max(300, int(len(item_dict) * 0.01 * num_negs))
    negs_u = call_get_negs_by_lsh(sample_num_u, user_dict)
    negs_v = call_get_negs_by_lsh(sample_num_v, item_dict)
    return negs_u,negs_v

def call_get_negs_by_lsh(sample_num, obj_dict):                                                                         # obj_dict就是输入的user或item的双重嵌套字典
    lsh_0, lsh_5, keys, ms = construct_lsh(obj_dict)
    visited = []
    negs_dict = {}
    for i in range(len(keys)):
        record = []
        if i in visited:
            continue
        visited.append(i)
        record.append(i)
        total_list = set(keys)
        sim_list = set(lsh_0.query(ms[i]))
        high_sim_list = set(lsh_5.query(ms[i]))
        total_list = list(total_list - sim_list)
        for j in high_sim_list:
            total_list = set(total_list)
            ind = keys.index(j)
            if ind not in visited:
                visited.append(ind)
                record.append(ind)
            sim_list_child = set(lsh_0.query(ms[ind]))
            total_list = list(total_list - sim_list_child)
        total_list = random.sample(list(total_list), min(sample_num, len(total_list)))
        for j in record:
            key = keys[j]
            negs_dict[key] = total_list
    return negs_dict

def construct_lsh(obj_dict):                                                                                            # obj_dict就是输入的user或item的双重嵌套字典
    # 下列MinHashLSH()中参数的含义分别为：
    # threshold表示jaccard距离阈值的设定，默认为0.9。其中两集合X和Y之间的jaccard相似度计算方法为 |X交Y|/|X并Y|；
    # num_perm是哈希置换函数设定个数，在weighted-MinHash中为样本规模的大小
    # params表示bands的数量与规模的大小
    # weights在这里没有出现，它表示优化jaccard阈值，能够弹性选择
    lsh_0 = MinHashLSH(threshold = 0, num_perm = 128, params = None)                                                    # https://blog.csdn.net/IOT_victor/article/details/104044453 ，是一种计算相似程度的近似算法
    lsh_5 = MinHashLSH(threshold = 0.6, num_perm = 128, params = None)                                                  # MinHashLSH中的LSH是“本地敏感哈希（LSH：Locality Sensitive Hashing ）索引”
    # forest = MinHashLSHForest(num_perm=128)
    keys = list(obj_dict.keys())                                                                                        # 这两处确实要转换为list
    values = list(obj_dict.values())                                                                                    # 输出的是一个list，其中的元素是dict，即形如[{item1:rating1, item2: rating2}, {item3:rating3}...]这样，即为这个嵌套字典中的后面部分
    ms = []
    for i in range(len(keys)):
        temp = MinHash(num_perm = 128)                                                                                  # MinHash()是最小哈希
        for d in values[i]:                                                                                             # 如果这个列表中的元素是个字典且对应多个value，那么当i走到相应位置时，会将其所有对应的值一同输出
            temp.update(d.encode('utf8'))

        ms.append(temp)
        lsh_0.insert(keys[i], temp)
        lsh_5.insert(keys[i], temp)

    return lsh_0,lsh_5, keys, ms