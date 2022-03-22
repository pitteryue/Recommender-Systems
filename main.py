import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as Data                                                                                         # 里面有minibatch实现所需要的DataLoader
from torch.autograd import Variable
from sklearn.utils import shuffle
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import numpy as np
from sklearn import preprocessing
from BiNE_graph_utils import GraphUtils
import random
import math
import os
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, auc, precision_recall_fscore_support
from copy import deepcopy

from data import *
from prediction import *
from evaluation import *
from UBCF_implicit import *
from UBCF_explicit import *
from PhysRec import *
from FunkSVD import *
from PMF import *
from SAE import *
from NCF import *
from FM2 import *
from BiNE_graph import *
from BiNE_graph_utils import *
from BiNE_lsh import *
from TransE import *

from TryOne import *

# import surprise                 # 一个专门的 recommender system 包
# import xlearn                   # 一个专门的 FM 家族的包

def OverallAverage_main():
    total_rating = 0.0
    count_train = 0.0
    for row in train_dataset.iloc():
        total_rating += row['rating']
        count_train += 1.0
    overall_avg = total_rating/count_train

    total_MAE = 0.0
    total_RMSE = 0.0
    count_test = 0.0
    for row in test_dataset.iloc():
        total_MAE += abs(row['rating'] - overall_avg)
        total_RMSE += (row['rating'] - overall_avg)**2
        count_test += 1.0
    MAE = total_MAE / count_test
    RMSE = math.sqrt(total_MAE / count_test)
    print('MAE:', MAE, 'RMSE:', RMSE)

def UBCF_explicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)

    ubcf = UBCF_explicit("pearsonr" , train_dataset_rating_matrix, K)
    similarity_matrix = ubcf.similarity_matrix()
    similarity_matrix[similarity_matrix < 0] = 0                                                                        # 因为后面计算评分时，分母必须是相似性绝对值的和（注意，分子的相对程度不是奥！），所以这里要么转换为绝对值，要么将负的视为0（源代码用的后面这个方法）

    estimated_rating_matrix = ubcf.prediction(similarity_matrix)                                                        # 这里的estimated_rating_matrix还没有进行-9999处理
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def IBCF_explicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)
    train_dataset_rating_matrix_T = pd.DataFrame(train_dataset_rating_matrix.values.T,index=train_dataset_rating_matrix.columns.values,columns=train_dataset_rating_matrix.index.values)

    ibcf = UBCF_explicit("pearsonr", train_dataset_rating_matrix_T, K)
    similarity_matrix = ibcf.similarity_matrix()
    similarity_matrix[similarity_matrix < 0] = 0

    # 再算评分矩阵
    estimated_rating_matrix_T = ibcf.prediction(similarity_matrix)
    estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_T.values.T,index=estimated_rating_matrix_T.columns.values,columns=estimated_rating_matrix_T.index.values)
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t', HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def Hybrid_explicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)
    train_dataset_rating_matrix_T = pd.DataFrame(train_dataset_rating_matrix.values.T,index = train_dataset_rating_matrix.columns.values, columns = train_dataset_rating_matrix.index.values)

    ubcf = UBCF_explicit("pearsonr", train_dataset_rating_matrix, K)
    similarity_matrix_ubcf = ubcf.similarity_matrix()
    similarity_matrix_ubcf[similarity_matrix_ubcf < 0] = 0
    estimated_rating_matrix_ubcf = ubcf.prediction(similarity_matrix_ubcf)

    ibcf = UBCF_explicit("pearsonr", train_dataset_rating_matrix_T, K)
    similarity_matrix_ibcf = ibcf.similarity_matrix()
    similarity_matrix_ibcf[similarity_matrix_ibcf < 0] = 0
    estimated_rating_matrix_ibcf_T = ibcf.prediction(similarity_matrix_ibcf)
    estimated_rating_matrix_ibcf = pd.DataFrame(estimated_rating_matrix_ibcf_T.values.T,index=estimated_rating_matrix_ibcf_T.columns.values,columns=estimated_rating_matrix_ibcf_T.index.values)

    estimated_rating_matrix = (estimated_rating_matrix_ubcf + estimated_rating_matrix_ibcf)/2
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def UBCF_implicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)

    ubcf = UBCF_implicit("pearsonr", train_dataset_rating_matrix, K)
    similarity_matrix = ubcf.similarity_matrix()
    similarity_matrix[similarity_matrix < 0] = 0

    estimated_rating_matrix = ubcf.prediction(similarity_matrix)
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

# IBCF直接继承UBCF，只不过输入转置一下，输出再转置回去
def IBCF_implicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)
    train_dataset_rating_matrix_T = pd.DataFrame(train_dataset_rating_matrix.values.T,index=train_dataset_rating_matrix.columns.values,columns=train_dataset_rating_matrix.index.values)

    ibcf = UBCF_implicit("pearsonr", train_dataset_rating_matrix_T, K)
    similarity_matrix = ibcf.similarity_matrix()
    similarity_matrix[similarity_matrix < 0] = 0

    estimated_rating_matrix_T = ibcf.prediction(similarity_matrix)
    estimated_rating_matrix = pd.DataFrame(estimated_rating_matrix_T.values.T,index=estimated_rating_matrix_T.columns.values,columns=estimated_rating_matrix_T.index.values)
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t', HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

# 源代码就是用UBCF和IBCF的结果求和后除以2
def Hybrid_implicit_main(K):
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)
    train_dataset_rating_matrix_T = pd.DataFrame(train_dataset_rating_matrix.values.T,index = train_dataset_rating_matrix.columns.values, columns = train_dataset_rating_matrix.index.values)

    ubcf = UBCF_implicit("pearsonr", train_dataset_rating_matrix, K)
    similarity_matrix_ubcf = ubcf.similarity_matrix()
    similarity_matrix_ubcf[similarity_matrix_ubcf < 0] = 0
    estimated_rating_matrix_ubcf = ubcf.prediction(similarity_matrix_ubcf)

    ibcf = UBCF_implicit("pearsonr", train_dataset_rating_matrix_T, K)
    similarity_matrix_ibcf = ibcf.similarity_matrix()
    similarity_matrix_ibcf[similarity_matrix_ibcf < 0] = 0
    estimated_rating_matrix_ibcf_T = ibcf.prediction(similarity_matrix_ibcf)
    estimated_rating_matrix_ibcf = pd.DataFrame(estimated_rating_matrix_ibcf_T.values.T,index=estimated_rating_matrix_ibcf_T.columns.values,columns=estimated_rating_matrix_ibcf_T.index.values)

    estimated_rating_matrix = (estimated_rating_matrix_ubcf + estimated_rating_matrix_ibcf)/2
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def HC_main():
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)

    physrec = PhysRS(train_dataset_rating_matrix)
    estimated_rating_matrix = physrec.HC_propogation()                                                                  # 这里返回的已经是一个DataFrame类型了
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset, valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def MD_main():
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)

    physrec = PhysRS(train_dataset_rating_matrix)
    estimated_rating_matrix = physrec.MD_propogation()                                                                  # 这里返回的已经是一个DataFrame类型了
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

def HC_MD_main():
    train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)

    physrec = PhysRS(train_dataset_rating_matrix)
    estimated_rating_matrix = physrec.HC_MD_propogation(0.5)                                                            # 这里的融合参数lambda是针对HC和MD模型中的两个k(item)而言的
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending = False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
        del recommendation_list

# 这里为FunkSVD使用"早停法"防止过拟合
def FunkSVD_main(max_epoch, early_stopping,learning_rate):
    funksvd = FunkSVD(true_train_dataset, validation_dataset, user_embeddings, item_embeddings, n_latent, max_epoch,learning_rate)
    best_user_embeddings, best_item_embeddings, best_epoch = funksvd.train(early_stopping)

    # 预测
    prediction = PREDICTION(true_train_dataset)
    estimated_rating_matrix = prediction.GET_ESTIMATED_RATING_MATRIX(best_user_embeddings, best_item_embeddings)

    valid_items = estimated_rating_matrix.columns                                                                       # 获取DataFrame列名(这里也就是item的标签)
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):                                                                   # 返回行数
        row = estimated_rating_matrix.iloc[i]                                                                           # 遍历行，按列输出其Name(即索引)所对应的 列名(号) 和 元素值，这样一来就成为了Series类型。特别注意，将行号(从0开始)与索引区分开，每个行号对应着一个索引
        user_id = row.name                                                                                              # Series.name读取其在DataFrame相应位置中的index
        items = row.sort_values(ascending=False).index                                                                  # 对Series按从高到低的顺序排序，这样就能获取其对应的索引了，返回索引和值，最后.index提取其中的索引
        all_recommendation_list[user_id] = items

    # 评测
    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

def TryOne_main(max_epoch, early_stopping, learning_rate):

    tryone = TryOne(true_train_dataset, validation_dataset, user_embeddings, n_latent, max_epoch,learning_rate,R)
    best_user_embeddings, best_epoch = tryone.train(early_stopping)

    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t', HR, '\t', NDCG, '\t', MAE, '\t', RMSE)

        del recommendation_list

# 这里为PMF借助贝叶斯先验来确定正则化系数以防止过拟合
def PMF_main(max_epoch, early_stopping,learning_rate):
    pmf = PMF(true_train_dataset, validation_dataset, user_embeddings, item_embeddings, n_latent, max_epoch,learning_rate)
    best_user_embeddings, best_item_embeddings, best_epoch = pmf.train(early_stopping)

    # 进行预测
    prediction = PREDICTION(true_train_dataset)
    estimated_rating_matrix = prediction.GET_ESTIMATED_RATING_MATRIX(best_user_embeddings, best_item_embeddings)

    valid_items = estimated_rating_matrix.columns                                                                       # 获取DataFrame列名(这里也就是item的标签)
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):                                                                   # 返回行数
        row = estimated_rating_matrix.iloc[i]                                                                           # 遍历行，按列输出其Name(即索引)所对应的 列名(号) 和 元素值，这样一来就成为了Series类型。特别注意，将行号(从0开始)与索引区分开，每个行号对应着一个索引
        user_id = row.name                                                                                              # Series.name读取其在DataFrame相应位置中的index
        items = row.sort_values(ascending=False).index                                                                  # 对Series按从高到低的顺序排序，这样就能获取其对应的索引了，返回索引和值，最后.index提取其中的索引
        all_recommendation_list[user_id] = items

    # 评测
    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

def FM2_main(max_epoch, early_stopping,learning_rate):
    fm2 = FM2(true_train_dataset, validation_dataset, user_embeddings, item_embeddings, n_latent, max_epoch,learning_rate)
    best_w0, best_w_user, best_w_item, best_user_embeddings, best_item_embeddings, best_epoch = fm2.train(early_stopping)

    # 准备预测评分矩阵
    max_user = max(true_train_dataset['user'])
    max_item = max(true_train_dataset['item'])
    estimated_rating_matrix = pd.DataFrame(np.zeros((max_user + 1, max_item + 1)))
    for row in true_train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999
    estimated_rating_matrix = estimated_rating_matrix.loc[~(estimated_rating_matrix == 0).all(axis=1)]
    estimated_rating_matrix = estimated_rating_matrix.loc[:, (estimated_rating_matrix != 0).any(axis=0)]

    # 计算预测评分矩阵
    for user_index, user_embedding in best_user_embeddings.items():
        for item_index, item_embedding in best_item_embeddings.items():
            if estimated_rating_matrix.loc[user_index, item_index] != -9999:
                interaction_1 = user_embedding + item_embedding
                interaction_2 = np.multiply(user_embedding, user_embedding) + np.multiply(item_embedding,item_embedding)
                interaction = np.sum(np.multiply(interaction_1, interaction_1) - interaction_2) / 2
                y = best_w0 + (best_w_user[int(user_index)] + best_w_item[int(item_index)]) + interaction
                estimated_rating_matrix.loc[user_index, item_index] = y

    valid_items = estimated_rating_matrix.columns                                                                       # 获取DataFrame列名(这里也就是item的标签)
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):                                                                   # 返回行数
        row = estimated_rating_matrix.iloc[i]                                                                           # 遍历行，按列输出其Name(即索引)所对应的 列名(号) 和 元素值，这样一来就成为了Series类型。特别注意，将行号(从0开始)与索引区分开，每个行号对应着一个索引
        user_id = row.name                                                                                              # Series.name读取其在DataFrame相应位置中的index
        items = row.sort_values(ascending=False).index                                                                  # 对Series按从高到低的顺序排序，这样就能获取其对应的索引了，返回索引和值，最后.index提取其中的索引
        all_recommendation_list[user_id] = items

    # 评测
    prediction = PREDICTION(true_train_dataset)
    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

def SAE_main(max_epoch, early_stopping,learning_rate):                                                                                # 只关注target>0的，训练集中等于0的那些不能看，参见矩阵分解方法就一目了然了！
    sae = SAE(item_size, n_latent)
    criterion = nn.MSELoss()                                                                                            # MSE是均方误差，RMSE是均方根误差
    optimizer = optim.RMSprop(sae.parameters(), lr = learning_rate,weight_decay = 0)                                             # 全称为root mean square prop，是AdaGrad算法的一种改进。weight_decay是一种防止过拟合的手段，和momentum作用的位置一样

    # 训练模型
    min_validation_error = np.inf
    best_epoch = 0
    error_counter = 0
    for _ in range(max_epoch):
        error_counter += 1
        output_train = sae(input_train)
        output_train[target_train == 0] = 0
        output_train_new = output_train[target_train != 0]
        target_train_new = target_train[target_train != 0]

        sae.zero_grad()
        train_loss = criterion(output_train_new, target_train_new)                                                              # 不能用总数做分母，否则Loss总体偏小，会极大影响迭代收敛方向的
        train_loss.backward()                                                                                           # 这里使用了optimizer.zero_grad() 反而收敛更慢，并且增大lr会在大误差下震荡，why？？？
        optimizer.step()
        sae.eval()

        output_validation = sae(input_validation)
        output_validation[target_validation ==0] = 0
        output_validation_new = output_validation[target_validation != 0 ]
        target_validation_new = target_validation[target_validation != 0 ]
        validation_loss = criterion(output_validation_new, target_validation_new)
        #print('Training loss:', train_loss.item(), 'Validation loss', validation_loss.item())

        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(sae, 'best_sae_model.pkl')
            best_epoch = _
            error_counter = 0
        if error_counter >= early_stopping:
            break

    best_sae = torch.load('best_sae_model.pkl')
    best_sae.eval()

    estimated_rating_matrix = deepcopy(true_train_dataset_rating_matrix)
    for row in estimated_rating_matrix.iloc():
        input = torch.tensor(row.values, dtype = torch.float32)
        output = best_sae(input).detach().numpy()
        estimated_rating_matrix.loc[row.name] = output
    for row in train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):                                                                   # .shape[0]是输出行数，而.shape[1]是输出列数
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    prediction = PREDICTION(train_dataset)
    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

# 不设置负样本的话，结果数据很难看
def GMF_main(max_epoch, early_stopping,learning_rate):

    GMF_model = NCF(user_num, item_num, n_latent, 'GMF')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(GMF_model.parameters(), lr=learning_rate)

    min_validation_error = np.inf
    best_epoch = 0
    error_counter = 0
    for _ in range(max_epoch):

        error_counter += 1
        output_train = GMF_model(user_train, item_train).unsqueeze(1)                                                   # 可以用.size()输出维度，如果不加，则为[x]而非[x,1]！
        train_loss = criterion(output_train,target_train)
        GMF_model.zero_grad()                                                                                           # 在使用.backward()之前要将梯度清零，否则会得到累加值。但之前的SAE和FM并没有这样,如果不这样，发散的话会更快，数字更大！
        train_loss.backward()                                                                                           # 这里是整体输入的，所以就要像原代码那样train_loss += loss, count 啥的了，那时一条条记录挨个输入的做法！
        optimizer.step()

        GMF_model.eval()                                                                                                # 验证集时要关掉dropout
        output_validation = GMF_model(user_validation, item_validation).unsqueeze(1)
        validation_loss = criterion(output_validation, target_validation)                                               # 注意防止的顺序，否则会出现负值
        #print('Training loss:', train_loss.item(), 'Validation loss:', validation_loss.item())

        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(GMF_model, 'best_GMF_model.pkl')
            best_epoch = _
            error_counter = 0
        if error_counter>= early_stopping:
            break

    best_GMF_model = torch.load('best_GMF_model.pkl')
    best_GMF_model.eval()

    # 下面是算每个用户对每个物品的预测得分，实现细节上我灵机一动，一改以往的挨个元素遍历，借助tensor快速批处理的优势，以向量为单位的视角进行输入！啊！我真是个小天才！
    estimated_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(true_train_dataset)
    row_size = estimated_rating_matrix.shape[0]
    user_rc = torch.tensor(estimated_rating_matrix.index.values.tolist()).unsqueeze(1).long()                           # 准备着与item_rc呼应，同时作为模型的批量输入

    prediction = PREDICTION(true_train_dataset)
    columns_set = estimated_rating_matrix.columns.values.tolist()
    for i in range(len(columns_set)):                                                                                   # 这里选择以列为单位，因为更新列比较容易一些
        item_rc = torch.tensor([columns_set[i] for size in range(row_size)]).unsqueeze(1)
        pred = best_GMF_model(user_rc, item_rc).tolist()
        estimated_rating_matrix[columns_set[i]] = pred

    # 最后进行一下排序前的-9999处理，这里或许还有除了挨个遍历外更好的方法(这一部分始终有些耗时间哦)
    for row in true_train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns                                                                       # 获取DataFrame列名(这里也就是item的标签)
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):                                                                   # 返回行数
        row = estimated_rating_matrix.iloc[i]                                                                           # 遍历行，按列输出其Name(即索引)所对应的 列名(号) 和 元素值，这样一来就成为了Series类型。特别注意，将行号(从0开始)与索引区分开，每个行号对应着一个索引
        user_id = row.name                                                                                              # Series.name读取其在DataFrame相应位置中的index
        items = row.sort_values(ascending=False).index                                                                  # 对Series按从高到低的顺序排序，这样就能获取其对应的索引了，返回索引和值，最后.index提取其中的索引
        all_recommendation_list[user_id] = items

    print(best_epoch)
    # 评测
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

# 由于之前将MLP放到CPU上跑会卡机，因此这里放到GPU上跑
def MLP_main(max_epoch, early_stopping,learning_rate):

    MLP_model = NCF(user_num, item_num, n_latent, 'MLP')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(MLP_model.parameters(), lr=learning_rate)

    min_validation_error = np.inf
    best_epoch = 0
    error_counter = 0
    for _ in range(max_epoch):
        error_counter += 1
        output_train = MLP_model(user_train, item_train).unsqueeze(1)
        train_loss = criterion(output_train, target_train)
        MLP_model.zero_grad()
        train_loss.backward()
        optimizer.step()

        MLP_model.eval()
        output_validation = MLP_model(user_validation, item_validation).unsqueeze(1)
        validation_loss = criterion(output_validation, target_validation)
        # print('Training loss:', train_loss.item(), 'Validation loss:', validation_loss.item())

        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(MLP_model, 'best_MLP_model.pkl')
            best_epoch = _
            error_counter = 0
        if error_counter >= early_stopping:
            break

    best_MLP_model = torch.load('best_MLP_model.pkl')
    best_MLP_model.eval()

    # 下面是算每个用户对每个物品的预测得分，实现细节上我灵机一动，一改以往的挨个元素遍历，借助tensor快速批处理的优势，以向量为单位的视角进行输入！啊！我真是个小天才！
    estimated_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(true_train_dataset)
    row_size = estimated_rating_matrix.shape[0]
    user_rc = torch.tensor(estimated_rating_matrix.index.values.tolist()).unsqueeze(1).long()

    prediction = PREDICTION(true_train_dataset)
    columns_set = estimated_rating_matrix.columns.values.tolist()
    for i in range(len(columns_set)):
        item_rc = torch.tensor([columns_set[i] for size in range(row_size)]).unsqueeze(1)
        pred = best_MLP_model(user_rc, item_rc).tolist()
        estimated_rating_matrix[columns_set[i]] = pred

    # 最后进行一下排序前的-9999处理，这里或许还有除了挨个遍历外更好的方法(这一部分始终有些耗时间哦)
    for row in true_train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    print(best_epoch)
    # 评测
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

# 本地跑这个模型，n_latent = 32或许就是极限了，内存爆表！
def NeuMF_main(max_epoch, early_stopping,learning_rate):

    # GMF模型预训练
    GMF_model = NCF(user_num, item_num, n_latent, 'GMF')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(GMF_model.parameters(), lr=0.01)
    min_validation_error = np.inf
    error_counter = 0
    GMF_pretrain_best_epoch = 0
    for _ in range(max_epoch):
        error_counter += 1
        output_train = GMF_model(user_train, item_train).unsqueeze(1)
        train_loss = criterion(output_train, target_train)
        GMF_model.zero_grad()
        train_loss.backward()
        optimizer.step()
        GMF_model.eval()
        output_validation = GMF_model(user_validation, item_validation).unsqueeze(1)
        validation_loss = criterion(output_validation, target_validation)
        # print('Pre-Train GMF loss:', train_loss.item(), 'Validation loss:', validation_loss.item())
        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(GMF_model, 'best_GMF_model.pkl')
            GMF_pretrain_best_epoch = _
            error_counter = 0
        if error_counter >= early_stopping:
            break
    best_GMF_model = torch.load('best_GMF_model.pkl')
    best_GMF_model.eval()

    # MLP模型预训练
    MLP_model = NCF(user_num, item_num, n_latent, 'MLP')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(MLP_model.parameters(), lr=0.01)
    min_validation_error = np.inf
    error_counter = 0
    MLP_pretrain_best_epoch = 0
    for _ in range(max_epoch):
        error_counter += 1
        output_train = MLP_model(user_train, item_train).unsqueeze(1)
        train_loss = criterion(output_train, target_train)
        MLP_model.zero_grad()
        train_loss.backward()
        optimizer.step()
        MLP_model.eval()
        output_validation = MLP_model(user_validation, item_validation).unsqueeze(1)
        validation_loss = criterion(output_validation, target_validation)
        # print('Pre-Train MLP loss:', train_loss.item(), 'Validation loss:', validation_loss.item())
        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(MLP_model, 'best_MLP_model.pkl')
            MLP_pretrain_best_epoch = _
            error_counter = 0
        if error_counter >= early_stopping:
            break
    best_MLP_model = torch.load('best_MLP_model.pkl')
    best_MLP_model.eval()

    # 开始最终的NeuMF模型训练
    NeuMF_model = NCF(user_num, item_num, n_latent, 'NeuMF', best_GMF_model, best_MLP_model)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(NeuMF_model.parameters(), lr=learning_rate)
    min_validation_error = np.inf
    NCF_best_epoch = 0
    error_counter = 0
    for _ in range(max_epoch):
        error_counter += 1
        output_train = NeuMF_model(user_train, item_train).unsqueeze(1)
        train_loss = criterion(output_train, target_train)
        NeuMF_model.zero_grad()
        train_loss.backward()
        optimizer.step()
        NeuMF_model.eval()
        output_validation = NeuMF_model(user_validation, item_validation).unsqueeze(1)
        validation_loss = criterion(output_validation, target_validation)
        # print('Training loss:', train_loss.item(), 'Validation loss:', validation_loss.item())
        if validation_loss.item() < min_validation_error:
            min_validation_error = validation_loss.item()
            torch.save(NeuMF_model, 'best_NeuMF_model.pkl')
            NCF_best_epoch = _
            error_counter = 0
        if error_counter >= early_stopping:
            break
    best_NeuMF_model = torch.load('best_NeuMF_model.pkl')
    best_NeuMF_model.eval()

    # 做预测算指标咯
    estimated_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(true_train_dataset)
    row_size = estimated_rating_matrix.shape[0]
    user_rc = torch.tensor(estimated_rating_matrix.index.values.tolist()).unsqueeze(1).long()

    prediction = PREDICTION(true_train_dataset)
    columns_set = estimated_rating_matrix.columns.values.tolist()
    for i in range(len(columns_set)):
        item_rc = torch.tensor([columns_set[i] for size in range(row_size)]).unsqueeze(1)
        pred = best_NeuMF_model(user_rc, item_rc).tolist()
        estimated_rating_matrix[columns_set[i]] = pred
    for row in true_train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = -9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=False).index
        all_recommendation_list[user_id] = items

    print(GMF_pretrain_best_epoch, '\t' , MLP_pretrain_best_epoch, '\t' ,NCF_best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t',  HR, '\t' ,NDCG, '\t' ,MAE, '\t' ,RMSE)

        del recommendation_list

def TransE_main(max_epoch, early_stopping,learning_rate):
    transE = TransE(user_list, item_list, relation_list, triplet_list, 1 , n_latent,learning_rate)                                    # 倒数第二个参数是 margin
    transE.initialize()
    best_user_embeddings, best_item_embeddings, best_epoch = transE.transE(max_epoch,early_stopping, validation_dataset)

    # 预测，因为对两两user-item对求距离是个双for循环，很耗时，所以这里转换为矩阵运算，也类似于FM中将一个双for转换为多个单for的技巧
    prediction = PREDICTION(true_train_dataset)
    user_embeddings_matrix = pd.DataFrame.from_dict(best_user_embeddings,orient='index')
    item_embeddings_matrix = pd.DataFrame.from_dict(best_item_embeddings)
    estimated_rating_matrix = user_embeddings_matrix.dot(item_embeddings_matrix)
    user_square = deepcopy(estimated_rating_matrix)
    for user in user_square.index.values:
        user_square.loc[user,:] = sum([i**2 for i in best_user_embeddings[user]])
    item_square = deepcopy(estimated_rating_matrix)
    for item in item_square.columns.values:
        item_square.loc[:,item] = sum([i**2 for i in best_item_embeddings[item]])
    estimated_rating_matrix = -2 * estimated_rating_matrix + user_square + item_square
    for row in true_train_dataset.iloc():
        estimated_rating_matrix.loc[row['user'], row['item']] = 9999

    valid_items = estimated_rating_matrix.columns
    all_recommendation_list = {}
    for i in range(estimated_rating_matrix.shape[0]):
        row = estimated_rating_matrix.iloc[i]
        user_id = row.name
        items = row.sort_values(ascending=True).index                                                                   # 这里也要改一下！因为TransE背景下距离越小，节点越相似哦！
        all_recommendation_list[user_id] = items

    # 评测
    print(best_epoch)
    for recommendation_size in range(10, 60, 10):
        recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
        evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, true_train_dataset,valid_items, test_dataset)
        Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
        print(Precision, '\t', HR, '\t', NDCG, '\t', MAE, '\t', RMSE)

        del recommendation_list

def BiNE_walk_generator(gul,args):
    print("calculate centrality...")
    gul.calculate_centrality(args.mode)                                                                                 # mode默认是hits，得到的是gul中的private parameters（字典） 即 self.authority_u 和 self.authority_v = {}, {}
    if args.large == 0:                                                                                                 # 默认为.large = 0
        gul.homogeneous_graph_random_walks(percentage = args.p, maxT = args.maxT, minT = args.minT)
    elif args.large == 1:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage = args.p, maxT = args.maxT, minT = args.minT)
    elif args.large == 2:
        gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile = args.train_data, percentage = args.p, maxT = args.maxT, minT = args.minT)

    return gul

def BiNE_get_context_and_negative_samples(gul, args):
    if args.large == 0:                                                                                                 # 默认为0
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)                                                                  # ns是 number of negative samples.
        print("negative samples is ok.....")                                                                            # 返回的也是嵌套dict，具体细节有些没搞明白，尤其是那个哈希？但可以先不管
        # ws 表示 window size，ns 表示 number of negative samples，G_u表示只包含user的同构图
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.G_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.G_v, gul.walks_v, args.ws, args.ns, neg_dict_v)
    else:
        neg_dict_u, neg_dict_v = gul.get_negs(args.ns)
        print("negative samples is ok.....")
        context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, args.ws, args.ns, neg_dict_u)
        context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, args.ws, args.ns, neg_dict_v)

    return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, gul.node_u, gul.node_v                               # gul.node_u 和 gul.node_v 是list类型的数据

def BiNE_init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args):
    for i in node_u:
        vectors = np.random.random([1, args.d])                                                                         # 表示生成1行、d列的随机浮点数，其范围在(0,1)之间，其中d是embedding size，默认为128
        help_vectors = np.random.random([1, args.d])
        node_list_u[i] = {}                                                                                             # node_list_u 是一个嵌套dict，因为你每个node有embedding vectors和context vectors
        # preprocessing是sklearn的函数，norm可以为为l1(样本各个特征值除以各个特征值的绝对值之和)、l2(样本各个特征值除以各个特征值的平方之和)或max(样本各个特征值除以样本中特征值最大的值)，默认为l2
        node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm = 'l2')
        node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm = 'l2')
    for i in node_v:
        vectors = np.random.random([1, args.d])
        help_vectors = np.random.random([1, args.d])
        node_list_v[i] = {}
        node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm = 'l2')
        node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm = 'l2')

    return node_list_u, node_list_v

def BiNE_skip_gram(center, contexts, negs, node_list, lam, pa):                                                              # 分别对应：u, z, neg_u, node_list_u, lam, alpha, 其中z是context_u中一个一个进行输入的，对items训练也类似
    loss = 0
    I_z = {center: 1}                                                                                                   # indication function
    for node in negs:
        I_z[node] = 0
    V = np.array(node_list[contexts]['embedding_vectors'])
    update = [[0] * V.size]
    for u in I_z.keys():
        if node_list.get(u) is  None:
            pass
        Theta = np.array(node_list[u]['context_vectors'])
        X = float(V.dot(Theta.T))
        sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
        update += pa * lam * (I_z[u] - sigmod) * Theta
        node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
        try:
            loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
        except:
            pass
    return update, loss

def BiNE_KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    loss = 0
    e_ij = edge_dict_u[u][v]

    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X = float(U.dot(V.T))

    sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))

    update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * V
    update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * U

    try:
        loss += gamma * e_ij * math.log(sigmod)
    except:
        pass
    return update_u, update_v, loss

def BiNE_train_by_sampling(train_dataset,test_dataset,args):
    print('======== experiment settings =========')
    alpha, beta, gamma, lam = args.alpha, args.beta, args.gamma, args.lam


    print("constructing graph....")
    gul = GraphUtils()
    gul.construct_training_graph(train_dataset)                                                                         # 这里创建的是二部图
    edge_dict_u = gul.edge_dict_u                                                                                       # 一个二层嵌套dict，可以通过dixt[user][item]检索出其对应的rating
    edge_list = gul.edge_list                                                                                           # 即[(user, item, rating), ...]
    BiNE_walk_generator(gul, args)                                                                                      # 这里应该返回个gul吧？之前源代码木有gul =，（想想看其实这里也不用返回吧，因为所有的更改，尤其是gul中对象walks的生成，直接就存储进去了）

    print("getting context and negative samples....")
    context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = BiNE_get_context_and_negative_samples(gul, args)

    print("============== training ==============")

    for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
        print('BiNE', 'n_latent=', n_latent)

        args.max_iter = 200
        args.d = n_latent
        node_list_u, node_list_v = {}, {}
        node_list_u, node_list_v = BiNE_init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, args)
        last_loss, count, epsilon = 0, 0, 1e-3

        for iter in range(0, args.max_iter):
            # s1 = "\r[%s%s]%0.2f%%" % ("*" * iter, " " * (args.max_iter - iter), iter * 100.0 / (args.max_iter - 1))# 这里实际上是要输出进度条
            loss = 0
            visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))  # node_list_u和node_list_v存储着每个user和item的embedding vectors
            visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))  # 这里的visited_u的输出是个字典，形如{u1:0, ...}，visited_v也是这样，每次迭代时字典的values清零
            random.shuffle(edge_list)  # 即gul中的一个对象，存储着edge_list_u_v的三元组
            for i in range(len(edge_list)):  # 对每条边都进行，
                u, v, w = edge_list[i]

                # 对users的embeddings进行训练
                length = len(context_dict_u[u])
                random.shuffle(context_dict_u[u])
                if visited_u.get(u) < length:
                    index_list = list(range(visited_u.get(u), min(visited_u.get(u) + 1,length)))  # range(start, stop)但不包含stop，这里看样子就是输出的visited_u.get(u)这一个位置的index
                    for index in index_list:
                        context_u = context_dict_u[u][index]
                        neg_u = neg_dict_u[u][index]
                        for z in context_u:
                            tmp_z, tmp_loss = BiNE_skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                            node_list_u[z]['embedding_vectors'] += tmp_z  # 这里嵌套字典的Key可以是字符串名
                            loss += tmp_loss
                    visited_u[u] = index_list[-1] + 3  # 本轮结束后光标移动到的位置

                # 对items的embeddings进行训练
                length = len(context_dict_v[v])
                random.shuffle(context_dict_v[v])
                if visited_v.get(v) < length:
                    index_list = list(range(visited_v.get(v), min(visited_v.get(v) + 1, length)))
                    for index in index_list:
                        context_v = context_dict_v[v][index]
                        neg_v = neg_dict_v[v][index]
                        for z in context_v:
                            tmp_z, tmp_loss = BiNE_skip_gram(v, z, neg_v, node_list_v, lam, beta)
                            node_list_v[z]['embedding_vectors'] += tmp_z  # 这也是skip_gram的方法，对它们进行求和
                            loss += tmp_loss
                    visited_v[v] = index_list[-1] + 3

                update_u, update_v, tmp_loss = BiNE_KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam,
                                                                  gamma)
                loss += tmp_loss
                node_list_u[u]['embedding_vectors'] += update_u
                node_list_v[v]['embedding_vectors'] += update_v

            delta_loss = abs(loss - last_loss)
            if last_loss > loss:
                lam *= 1.05
            else:
                lam *= 0.95
            last_loss = loss
            if delta_loss < epsilon:
                break
            # sys.stdout.write(s1)
            # sys.stdout.flush()

        # save_to_file(node_list_u, node_list_v, args)                                                               # 最终要得到的其实就是users和items的embedding vectors 和 context vectors
        print("")
        # 之后在推荐过程中就是将对应的user和item的embedding vectors做点积，就得到了将用于推荐排序的预测评分
        # 预测
        best_user_embeddings = {}
        best_item_embeddings = {}
        for key, value in node_list_u.items():
            best_user_embeddings[int(key[1:])] = value['embedding_vectors'].squeeze()  # 这里要对numpy数组压缩一个维度奥！
        for key, value in node_list_v.items():
            best_item_embeddings[int(key[1:])] = value['embedding_vectors'].squeeze()

        prediction = PREDICTION(train_dataset)
        estimated_rating_matrix = prediction.GET_ESTIMATED_RATING_MATRIX(best_user_embeddings, best_item_embeddings)
        valid_items = estimated_rating_matrix.columns
        all_recommendation_list = {}
        for i in range(estimated_rating_matrix.shape[0]):
            row = estimated_rating_matrix.iloc[i]
            user_id = row.name
            items = row.sort_values(ascending=False).index
            all_recommendation_list[user_id] = items
        for recommendation_size in range(10, 60, 10):
            recommendation_list = prediction.GET_RECOMMENDATION_LIST(recommendation_size, all_recommendation_list)
            evaluation = EVALUATION(estimated_rating_matrix, recommendation_size, recommendation_list, train_dataset,
                                    valid_items, test_dataset)
            Precision, HR, NDCG, MAE, RMSE = evaluation.MERICS_01()
            print(Precision, '\t', HR, '\t', NDCG, '\t', MAE, '\t', RMSE)
            del recommendation_list

def BiNE_main(max_epoch, early_stopping):
    parser = ArgumentParser("BiNE", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    # 'BiNE'是程序的名称；formatter_class用于自定义帮助文档输出格式的类，其中ArgumentDefaultsHelpFormatter表示自动添加默认的值的信息到每一个帮助信息的参数中；
    # conflict_handler解决冲突选项的策略（通常是不必要的），其中 'resolve' 值可以提供给 ArgumentParser 的 conflict_handler= 参数
    parser.add_argument('--train-data', default='rating_train.csv',help='Input graph file.')                            # 当用户请求帮助时（一般是通过在命令行中使用 -h 或 --help 的方式），这些 help 描述将随每个参数一同显示
    parser.add_argument('--test-data', default='rating_test.csv')                                                       # 对于name，在声明时最前方要加上--，而后面在调用时就像其他类那样用.即可
    parser.add_argument('--model-name', default='default', help='name of model.')
    parser.add_argument('--vectors-u', default='vectors_u.dat', help="file of embedding vectors of U")
    parser.add_argument('--vectors-v', default='vectors_v.dat', help="file of embedding vectors of V")
    parser.add_argument('--case-train', default='case_train.dat', help="file of training data for LR")
    parser.add_argument('--case-test', default='case_test.dat', help="file of testing data for LR")
    parser.add_argument('--ws', default=5, type=int,help='window size.')                                                # 因为parser会默认将传入的选项当做字符串，所以这里想要整型，则必须为它指定为int的type
    parser.add_argument('--ns', default=4, type=int, help='number of negative samples.')
    parser.add_argument('--d', default=128, type=int, help='embedding size.')                                           # 向量维度如果后面需要迭代，可以修改
    parser.add_argument('--maxT', default=32, type=int, help='maximal walks per vertex.')
    parser.add_argument('--minT', default=1, type=int, help='minimal walks per vertex.')
    parser.add_argument('--p', default=0.15, type=float, help='walk stopping probability.')
    parser.add_argument('--alpha', default=0.01, type=float, help='trade-off parameter alpha.')
    parser.add_argument('--beta', default=0.01, type=float, help='trade-off parameter beta.')
    parser.add_argument('--gamma', default=0.1, type=float, help='trade-off parameter gamma.')
    parser.add_argument('--lam', default=0.01, type=float, help='learning rate lambda.')
    parser.add_argument('--max-iter', default=max_epoch, type=int, help='maximal number of iterations.')
    parser.add_argument('--stop', default=early_stopping, type=int, help='early stopping number of iterations.')
    parser.add_argument('--top-n', default=10, type=int, help='recommend top-n items for each user.')
    parser.add_argument('--rec', default=1, type=int, help='calculate the recommendation metrics.')
    parser.add_argument('--lip', default=0, type=int, help='calculate the link prediction metrics.')
    parser.add_argument('--large', default=0, type=int,help='for large bipartite, 1 do not generate homogeneous graph file; 2 do not generate homogeneous graph， 这里的备注有问题')
    parser.add_argument('--mode', default='hits', type=str, help='metrics of centrality')
    args = parser.parse_args()                                                                                          # 将参数字符串转换为对象并将其设为命名空间的属性。 返回带有成员的命名空间
    BiNE_train_by_sampling(true_train_dataset,test_dataset,args)                                                                                        # 将命名空间中的参数全部进行传入

def construct_smples_for_GMF_MLP_NCF():
    validation_samples = validation_dataset                                                                             # 这些放在最外面，因为每次用到的都是一样的，以免重复操作
    user_validation = torch.tensor(validation_samples['user'].tolist()).unsqueeze(1).long()
    item_validation = torch.tensor(validation_samples['item'].tolist()).unsqueeze(1).long()
    target_validation = torch.tensor(validation_samples['rating'].tolist()).unsqueeze(1).float()
    user_num = max(train_dataset['user']) + 1
    item_num = max(train_dataset['item']) + 1
    true_train_samples = pd.concat([true_train_dataset, negative_samples]).sample(frac=1)  # 打乱操作
    user_train = torch.tensor(true_train_samples['user'].tolist()).unsqueeze(1).long()
    item_train = torch.tensor(true_train_samples['item'].tolist()).unsqueeze(1).long()
    target_train = torch.tensor(true_train_samples['rating'].tolist()).unsqueeze(1).float()
    # torch_dataset = Data.TensorDataset(user_train, item_train, target_train)                                          # 进行mini-batch操作
    # loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True,num_workers=0)

    return user_train, item_train, target_train, user_validation, item_validation, target_validation, user_num, item_num

def construct_samples_for_SAE():
    true_train_dataset_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)                              # 这里的validation要包含true_train和validation奥！
    true_train_dataset_rating_matrix.loc[:, :] = 0
    for row in true_train_dataset.iloc():                                                                               # 之前没有连续编号，因此会浪费很多空间
        true_train_dataset_rating_matrix.loc[row[0], row[1]] = row[2]
    input_samples_train = []                                                                                            # 输入采用整体tensor形式作为输入而不用for循环(否则慢得多)，则从list(因为好.append，试了几个numpy的追加元素方法，都不能按行追加)
    for row in true_train_dataset_rating_matrix.iloc():                                                                 # 这里再想办法把样本一次性弄成tensor后放到GPU上去)
        input_samples_train.append(row.values)                                                                          # 注意，torch模型的输入要是tensor哟！unsqueeze(0)后就会在0的位置加了一维就变成一行三列(1,3),起升维的作用
    input_train = torch.tensor(input_samples_train, dtype=torch.float32)
    target_train = input_train.clone()
    validation_dataset_rating_matrix = deepcopy(true_train_dataset_rating_matrix)
    for row in validation_dataset.iloc():
        validation_dataset_rating_matrix.loc[row[0], row[1]] = row[2]
    input_samples_validation = []
    for row in validation_dataset_rating_matrix.iloc():
        input_samples_validation.append(row.values)
    input_validation = torch.tensor(input_samples_validation, dtype=torch.float32)
    target_validation = input_validation.clone()

    return true_train_dataset_rating_matrix, input_train, target_train, validation_dataset_rating_matrix, input_validation, target_validation

def construct_samples_for_TransE():

    triplet_list = []
    relation_id = 0
    for i in range(len(true_train_dataset.index)):
        user = true_train_dataset.iloc[i,0]
        item = true_train_dataset.iloc[i,1]
        triplet_list.append((user, item, relation_id))
        relation_id += 1

    user_list = list(set(true_train_dataset.iloc[:,0].tolist()))
    item_list = list(set(true_train_dataset.iloc[:,1].tolist()))
    relation_list = list(range(relation_id))

    return triplet_list, user_list, item_list, relation_list

if __name__=='__main__':

    data_ratings = DATA_RATINGS(0.2)
    data = pd.read_table("Last.fm-90K.csv", sep=",", engine='python')
    data_ratings.RENUMBER_ITEM(data)
    '''
    # 随机划分数据并生成负样本
    data_ratings.TRAIN_TEST_BY_RANDOM('ratings.csv', 'train_ratings.csv','test_ratings.csv')
    dataset = pd.read_table("ratings.csv", sep=",", engine='python')
    train_dataset = pd.read_table("train_ratings.csv", sep=",", engine='python').sample(frac=1)
    train_rating_matrix = data_ratings.GET_TRAIN_RATING_MATRIX(train_dataset)
    data_ratings.OBTAIN_NEGATIVE_SAMPLES(train_dataset, train_rating_matrix)
    '''

    '''
    true_train_dataset = pd.read_table("ML-100k-train_ratings_random0.csv", sep=",", engine='python').sample(frac=1)                    # 读进来时打
    train_dataset = pd.read_table("ML-100k-train_ratings_random0.csv", sep=",", engine='python').sample(frac=1)
    validation_dataset = pd.read_table("ML-100k-test_ratings_random0.csv", sep=',', engine='python').sample(frac=1)
    test_dataset = pd.read_table("ML-100k-test_ratings_random0.csv", sep=',', engine='python').sample(frac=1)
    negative_samples = pd.read_table("ML-100k-negative_samples_random0.csv", sep=',', engine='python').sample(frac=1)

    # 对于后面要用到的validation_dataset，一定要先排除掉哪些在训练集中没有出现过的user和item所组成的项。（但train_dataset = true_train_dataset + validation_dataset这里不要筛除，因为本身就是为了还原train_dataset嘛）
    valid_users = set(true_train_dataset.iloc[:,0].tolist())
    valid_items = set(true_train_dataset.iloc[:,1].tolist())
    for row in validation_dataset.iloc():
        if row[0] not in valid_users or row[1] not in valid_items:
            validation_dataset = validation_dataset.drop(index = [row.name])
    print('total test_dataset size:', len(test_dataset.index),'valid test_dataset size:', len(validation_dataset.index))


    ############################################ 评分数据模型评测  ######################################################
    print('overall_average')
    OverallAverage_main()
    print('UBCF_explicit')
    UBCF_explicit_main(10)
    print('IBCF_explicit')
    IBCF_explicit_main(10)

    max_epoch = 100
    early_stopping = 20
    print(max_epoch, early_stopping)
    for _,learning_rate in enumerate([0.01]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('FunkSVD', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            FunkSVD_main(max_epoch, early_stopping, learning_rate)
    
    max_epoch = 100
    early_stopping = 20
    print(max_epoch, early_stopping)
    for _,learning_rate in enumerate([0.01]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('PMF', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            PMF_main(max_epoch, early_stopping,learning_rate)
    
    max_epoch = 100
    early_stopping = 20
    print(max_epoch, early_stopping)
    for _,learning_rate in enumerate([0.01]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('FM', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            FM2_main(max_epoch, early_stopping,learning_rate)

    max_epoch = 100000
    early_stopping = 100
    print(max_epoch, early_stopping)
    # 构造样本数据
    true_train_dataset_rating_matrix, input_train, target_train, validation_dataset_rating_matrix, input_validation, target_validation = construct_samples_for_SAE()
    item_size = true_train_dataset_rating_matrix.columns.size
    # 开始训练
    for _,learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('SAE', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            SAE_main(max_epoch, early_stopping,learning_rate)


    max_epoch = 100
    early_stopping = 20
    R = data_ratings.GET_TRAIN_RATING_MATRIX(true_train_dataset)
    print(max_epoch, early_stopping)
    for _, learning_rate in enumerate([0.01]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('TryOne', 'n_latent=', n_latent)
            user_embeddings = {}
            TryOne_main(max_epoch, early_stopping, learning_rate)


    ############################################ 01数据模型评测  ######################################################

    true_train_dataset = true_train_dataset[true_train_dataset['rating'] >= 3]
    validation_dataset = validation_dataset[validation_dataset['rating'] >= 3]
    train_dataset = train_dataset[train_dataset['rating'] >= 3]
    test_dataset = test_dataset[test_dataset['rating'] >= 3]
    true_train_dataset['rating'] = 1
    validation_dataset['rating'] = 1
    train_dataset['rating'] = 1
    test_dataset['rating'] = 1

    # 这里同样还需要进行一遍！之前没有注意到这一个bug，否则神经网络中依然会有无效的embedding被作为i验证集
    valid_users = set(true_train_dataset.iloc[:, 0].tolist())
    valid_items = set(true_train_dataset.iloc[:, 1].tolist())
    for row in validation_dataset.iloc():
        if row[0] not in valid_users or row[1] not in valid_items:
            validation_dataset = validation_dataset.drop(index=[row.name])
    print('total test_dataset size:', len(test_dataset.index), 'valid test_dataset size:',len(validation_dataset.index))

    
    print('UBCF_implicit')
    UBCF_implicit_main(10)
    print('IBCF_implicit')
    IBCF_implicit_main(10)
    print('Hybrid_implicit')
    Hybrid_implicit_main(10)

    print('HC')
    HC_main()
    print('MD')
    MD_main()
    print('HC_MD')
    HC_MD_main()
    
    max_epoch = 100000
    early_stopping = 100
    print(max_epoch, early_stopping)
    # 构造样本数据
    user_train, item_train, target_train, user_validation, item_validation, target_validation, user_num, item_num = construct_smples_for_GMF_MLP_NCF()
    # 开始训练
    for _,learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
        print(learning_rate)
        for i, n_latent in enumerate([4,8,16,32,64,128]):
            print('GMF', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            GMF_main(max_epoch, early_stopping,learning_rate)

    max_epoch = 100000
    early_stopping = 100
    print(max_epoch, early_stopping)
    # 构造样本数据
    user_train, item_train, target_train, user_validation, item_validation, target_validation, user_num, item_num = construct_smples_for_GMF_MLP_NCF()
    # 开始训练
    for _,learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('MLP', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            MLP_main(max_epoch, early_stopping,learning_rate)
    
    max_epoch = 100000
    early_stopping = 100
    print(max_epoch, early_stopping)
    #  构造样本数据
    user_train, item_train, target_train, user_validation, item_validation, target_validation, user_num, item_num = construct_smples_for_GMF_MLP_NCF()
    # 开始训练
    for _,learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('NeuMF', 'n_latent=', n_latent)
            user_embeddings = {}
            item_embeddings = {}
            NeuMF_main(max_epoch, early_stopping,learning_rate)                                                         # 注意，对预训练的lr在main中额外设置
    
    max_epoch = 100000
    early_stopping = 100
    print(max_epoch, early_stopping)
    # 构造样本数据
    triplet_list, user_list, item_list, relation_list = construct_samples_for_TransE()
    for _,learning_rate in enumerate([0.1, 0.05, 0.01, 0.005, 0.001]):
        print(learning_rate)
        for i, n_latent in enumerate([4, 8, 16, 32, 64, 128]):
            print('TransE', 'n_latent=', n_latent)
            TransE_main(max_epoch, early_stopping,learning_rate)

    
    for i in range(len(true_train_dataset.index)):
       true_train_dataset.iloc[i, 0] = ''.join(['u', str(true_train_dataset.iloc[i,0])])
       true_train_dataset.iloc[i, 1] = ''.join(['i', str(true_train_dataset.iloc[i,1])])
    sys.exit(BiNE_main(true_train_dataset,test_dataset))
    '''