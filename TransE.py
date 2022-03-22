from random import uniform, sample
from numpy import *
from copy import deepcopy
import numpy as np

class TransE:

    def __init__(self, userList, itemList, relationList, tripleList, margin, dim, learingRate, L1 = False):
        # 初始化是关于这俩东西的
        self.userList = userList                                                                                        # 一开始，entityList是entity的list；初始化后，变为字典，key是entity，values是其向量（使用narray）。
        self.itemList = itemList
        self.relationList = relationList                                                                                # 理由同上

        self.tripleList = tripleList                                                                                    # 该list中的元素为三元组tuple，结构为（h,t,r)，注意最后一个元素为r

        self.margin = margin
        self.dim = dim
        self.learingRate = learingRate

        self.loss = 0
        self.L1 = L1

    def initialize(self):
        userVectorList = {}                                                                                             # 老套路，为每个对象创建embeddings单层字典索引
        itemVectorList = {}
        relationVectorList = {}

        # 初始化user的embeddings
        for user in self.userList:
            n = 0
            userVector = []                                                                                             # 字典的value值也就是一个List
            while n < self.dim:                                                                                         # 逐元素初始化
                ram = uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))                                            # 按原论文进行的设置
                userVector.append(ram)
                n += 1
            userVector = norm(userVector)                                                                               # 用 1-norm 来进行归一化
            userVectorList[user] = userVector

        # 初始化user的embeddings
        for item in self.itemList:
            n = 0
            itemVector = []
            while n < self.dim:
                ram = uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))
                itemVector.append(ram)
                n += 1
            itemVector = norm(itemVector)
            itemVectorList[item] = itemVector

        # 初始化relation的embeddings
        for relation in self.relationList:
            n = 0
            relationVector = []
            while n < self.dim:
                ram = uniform(-6 / (self.dim ** 0.5), 6 / (self.dim ** 0.5))
                relationVector.append(ram)
                n += 1
            relationVector = norm(relationVector)
            relationVectorList[relation] = relationVector

        self.userList = userVectorList                                                                                  # 牛逼，这里直接把self变量类型都给变了
        self.itemList = itemVectorList
        self.relationList = relationVectorList

    def transE(self, max_epoch, early_stopping, validation_dataset):

        min_validation_error = np.inf
        error_counter = 0
        best_epoch = 0
        for cycleIndex in range(max_epoch):
            size = 1024                                                                                                  # 每次迭代训练时正样本数的抽样大小。感觉这里弄个比例比较好
            Sbatch = sample(self.tripleList, size)                                                                      # 每次迭代时不用所有的样本，而是从中随机抽样出一部分来进行训练
            Tbatch = []                                                                                                 # 元组对（原三元组，打碎的三元组）的列表 ：[((h,r,t),(h',r,t')), .....]
            # 对size个Sbatch中的每一个元组，都要为之生成一个corrupted triplet
            for sbatch in Sbatch:                                                                                       # 每次迭代都要进行该操作
                tripletWithCorruptedTriplet = (sbatch, self.getCorruptedTriplet(sbatch))
                if (tripletWithCorruptedTriplet not in Tbatch):                                                         # 为了避免重复（注意：一个原三元组可对应多个打碎的三元组，前提是这些打碎的三元组要不一样奥）
                    Tbatch.append(tripletWithCorruptedTriplet)                                                          # 是个列表，一对一对的，即正样本和负样本成一对

            # 开始训练，更新embeddings
            self.update(Tbatch)

            # 判断是否早停
            error_counter += 1
            validation_error = self.get_error_L1(validation_dataset)                                                    # 用的什么范数进行训练的，那么就用什么范数进行评估
            #print('Training RMSE: {:.4f} Validation RMSE: {:.4f}'.format(self.loss/size, validation_error))
            self.loss = 0                                                                                               # 这里注意清0奥，因为定义的是一个全局变量
            if validation_error < min_validation_error:
                min_validation_error = validation_error
                best_user_embeddings = deepcopy(self.userList)                                                          # 直接成群复制，python实在是太高效啦！
                best_item_embeddings = deepcopy(self.itemList)
                error_counter = 0
                best_epoch = cycleIndex
            if error_counter >= early_stopping:                                                                         # 即已经过了局部最优点了，且规定次数之类也没能再找到另一个更小的局部最小点（但可能被一些非全局最小点的局部最小点吸收了，跑了几次结果都不一样，有遇到这种情况）
                break

        return best_user_embeddings, best_item_embeddings, best_epoch

    def getCorruptedTriplet(self, triplet):
        i = uniform(-1, 1)
        if i < 0:                                                                                                       # 小于0，打坏三元组的第一项
            while True:
                userTemp = sample(self.userList.keys(), 1)[0]
                if userTemp != triplet[0]:                                                                            # 直到打碎后的这个entity与原三元组中的对应位置的entity不一样为止
                    break
            corruptedTriplet = (userTemp, triplet[1], triplet[2])
        else:                                                                                                           # 大于等于0，打坏三元组的第二项
            while True:
                itemTemp = sample(self.itemList.keys(), 1)[0]
                if itemTemp != triplet[1]:
                    break
            corruptedTriplet = (triplet[0], itemTemp, triplet[2])

        return corruptedTriplet

    def update(self, Tbatch):
        copyuserList = deepcopy(self.userList)
        copyitemList = deepcopy(self.itemList)
        copyRelationList = deepcopy(self.relationList)

        for tripletWithCorruptedTriplet in Tbatch:
            userEntityVector = copyuserList[tripletWithCorruptedTriplet[0][0]]                                          # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple，成对排列（所以不要打乱）
            itemEntityVector = copyitemList[tripletWithCorruptedTriplet[0][1]]
            relationVector = copyRelationList[tripletWithCorruptedTriplet[0][2]]
            userEntityVectorWithCorruptedTriplet = copyuserList[tripletWithCorruptedTriplet[1][0]]                      # 因为relation不会被corrupted，所以这里没必要再给出，直接用上面的就是了
            itemEntityVectorWithCorruptedTriplet = copyitemList[tripletWithCorruptedTriplet[1][1]]

            userEntityVectorBeforeBatch = self.userList[tripletWithCorruptedTriplet[0][0]]                              # 因为在迭代更新时需要用到上一轮的结果（第一轮时即用到初始化的）
            itemEntityVectorBeforeBatch = self.itemList[tripletWithCorruptedTriplet[0][1]]
            relationVectorBeforeBatch = self.relationList[tripletWithCorruptedTriplet[0][2]]
            userEntityVectorWithCorruptedTripletBeforeBatch = self.userList[tripletWithCorruptedTriplet[1][0]]
            itemEntityVectorWithCorruptedTripletBeforeBatch = self.itemList[tripletWithCorruptedTriplet[1][1]]

            if self.L1:
                distTriplet = distanceL1(userEntityVectorBeforeBatch, itemEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL1(userEntityVectorWithCorruptedTripletBeforeBatch, itemEntityVectorWithCorruptedTripletBeforeBatch, relationVectorBeforeBatch)
            else:
                distTriplet = distanceL2(userEntityVectorBeforeBatch, itemEntityVectorBeforeBatch, relationVectorBeforeBatch)
                distCorruptedTriplet = distanceL2(userEntityVectorWithCorruptedTripletBeforeBatch, itemEntityVectorWithCorruptedTripletBeforeBatch, relationVectorBeforeBatch)
            eg = self.margin + distTriplet - distCorruptedTriplet                                                       # 损失值，老规矩，接下来用于迭代更新！

            if eg > 0:                                                                                                  # 因为原文中的[function]+ 是一个取正值的函数
                self.loss += eg
                if self.L1:                                                                                             # 即 1-norm
                    tempPositive = itemEntityVectorBeforeBatch - userEntityVectorBeforeBatch - relationVectorBeforeBatch
                    tempNegtative = itemEntityVectorWithCorruptedTripletBeforeBatch - userEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch
                    tempPositiveL1 = []
                    tempNegtativeL1 = []
                    for i in range(self.dim):                                                                           # 不知道有没有pythonic的写法（比如列表推倒或者numpy的函数）？
                        if tempPositive[i] >= 0:
                            tempPositiveL1.append(self.learingRate)
                        else:
                            tempPositiveL1.append(-self.learingRate)

                        if tempNegtative[i] >= 0:
                            tempNegtativeL1.append(self.learingRate)
                        else:
                            tempNegtativeL1.append(-self.learingRate)

                    tempPositive = array(tempPositiveL1)                                                                # 直接覆盖掉！
                    tempNegtative = array(tempNegtativeL1)

                    # 更新正样本中的entity
                    userEntityVector = userEntityVector + tempPositive                                                  # 这里一定要分清，别弄混了！回归公式！！！
                    itemEntityVector = itemEntityVector - tempPositive
                    # 更新反样本中的entity
                    userEntityVectorWithCorruptedTriplet = userEntityVectorWithCorruptedTriplet - tempNegtative
                    itemEntityVectorWithCorruptedTriplet = itemEntityVectorWithCorruptedTriplet + tempNegtative
                    # 对于relation，不分正样本或反样本
                    relationVector = relationVector + tempPositive - tempNegtative                                      # 这里需要注意一下！

                else:
                    tempPositive = 2 * self.learingRate * (itemEntityVectorBeforeBatch - userEntityVectorBeforeBatch - relationVectorBeforeBatch)
                    tempNegtative = 2 * self.learingRate * (itemEntityVectorWithCorruptedTripletBeforeBatch - userEntityVectorWithCorruptedTripletBeforeBatch - relationVectorBeforeBatch)

                    # 更新正样本中的entity
                    userEntityVector = userEntityVector + tempPositive
                    itemEntityVector = itemEntityVector - tempPositive
                    # 更新反样本中的entity
                    userEntityVectorWithCorruptedTriplet = userEntityVectorWithCorruptedTriplet - tempNegtative
                    itemEntityVectorWithCorruptedTriplet = itemEntityVectorWithCorruptedTriplet + tempNegtative
                    # 对于relation，不分正样本或反样本
                    relationVector = relationVector + tempPositive  - tempNegtative

                # 只归一化这几个刚更新的向量，而不是按原论文那些一口气全更新了
                copyuserList[tripletWithCorruptedTriplet[0][0]] = norm(userEntityVector)                                # 括号里都是本轮迭代、本个Batch中得到更新后的嵌入向量
                copyitemList[tripletWithCorruptedTriplet[0][1]] = norm(itemEntityVector)
                copyuserList[tripletWithCorruptedTriplet[1][0]] = norm(userEntityVectorWithCorruptedTriplet)
                copyitemList[tripletWithCorruptedTriplet[1][1]] = norm(itemEntityVectorWithCorruptedTriplet)
                copyRelationList[tripletWithCorruptedTriplet[0][2]] = norm(relationVector)

        self.userList = copyuserList                                                                                    # 本质上其实就是更新节点和连边的嵌入向量
        self.itemList = copyitemList                                                                                    # 因为能保证每个样本不重样，因此可以等一轮迭代后统一进行更新
        self.relationList = copyRelationList

    def get_error_L1(self, validation_dataset):
        size = 0
        total_error = 0

        for row in validation_dataset.iloc():
            user = row[0]
            item = row[1]
            total_error += fabs(self.userList[user] - self.itemList[item]).sum()
            size += 1

        return total_error / size

    def get_error_L2(self,validation_dataset):
        size = 0
        total_error = 0

        for row in validation_dataset.iloc():
            user = row[0]
            item = row[1]
            s = self.userList[user] - self.itemList[item]
            total_error += np.sqrt(np.dot(s,s))
            size += 1

        return total_error/size

def norm(list):                                                                                                         # key下对应的嵌入向量是个list结构
    var = linalg.norm(list)                                                                                             # 这里可能有问题，现在的版本应该需要np.（参见手册）
    i = 0
    while i < len(list):
        list[i] = list[i] / var
        i += 1
    return array(list)

def distanceL1(h, t, r):
    s = h + r - t
    sum = fabs(s).sum()

    return sum

def distanceL2(h, t, r):
    s = h + r - t
    sum = (s * s).sum()

    return sum