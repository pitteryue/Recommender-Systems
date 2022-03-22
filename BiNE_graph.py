import logging                                                                                                          # logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from multiprocessing import cpu_count
import math

logger = logging.getLogger("deepwalk")                                                                                  #  指定name（即括号中的字符串），返回一个名称为name的Logger实例。如果再次使用相同的名字，是实例化一个对象。未指定name，返回Logger实例，名称是root，即根Logger。

class Graph(defaultdict):
  act = {}                                                                                                              # 意思应该是约定G是一个dict类型吧
  isWeight = False

  def __init__(self):
    super(Graph, self).__init__(list)                                                                                   # super(父类, self).方法名(参数，这里是python2的继承写法，其中Graph是父类)

  def make_consistent(self):
    t0 = time()                                                                                                         # 返回当时的时间戳
    if self.isWeight == True:
      for k in iterkeys(self):                                                                                          # 用于迭代dict中的keys吧
        self[k] = self.sortedDictValues(self[k])
        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
        self.remove_self_loops_dict()
    else:
      for k in iterkeys(self):
        self[k] = list(sorted(set(self[k])))
        t1 = time()
        logger.info('make_consistent: made consistent in {}s'.format(t1-t0))
        self.remove_self_loops()

    return self

  def remove_self_loops(self):
    removed = 0
    t0 = time()
    if self.isWeight == True:
      for x in self:
        if x in self[x].keys():
          del self[x][x]
          removed += 1
    else:
      for x in self:
        if x in self[x]:
          self[x].remove(x)
          removed += 1
    t1 = time()

    logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
    return self

  def nodes(self):
    return self.keys()

  def random_walk_restart(self, nodes, percentage, alpha = 0, rand = random.Random(), start = None):                    # nodes是一个list奥！rand = random.Random()用于生成随机数
    """ Returns a truncated random walk.
        percentage: probability of stopping walking
        alpha: probability of restarts.
        start: the start node of the random walk.
    """
    G = self
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(nodes)]                                                                                       # 这里的长度也是随机的？？？(choice()方法从一个列表，元组或字符串返回一个随机项)

    while len(path) < 1 or random.random() > percentage:                                                                # 前面这里len(path)<1应该有问题吧？？？（这里没问题，因为or后面的语句成立，也就继续进行，len(path)<1的目的是为了防止一个节点就停止随机游走的情况发生）or后面的语句体现了以概率percentage终止游走
      cur = path[-1]                                                                                                    # 将光标指向列表的最后一个元素
      if len(G[cur]) > 0:                                                                                               # 判断与path的最后一个元素相关联的边是否存在
        if rand.random() >= alpha:                                                                                      # alpha 表示 probability of restarts.
          add_node = rand.choice(G[cur])                                                                                # 掌握这种代码描述，即在node的邻居节点中选择一个继续游走
          while add_node == cur:                                                                                        # 这里还加了层保险，为了防止自环
            add_node = rand.choice(G[cur])
          path.append(add_node)
        else:
          path.append(path[0])                                                                                          # 即回到root node，开始重新随机游走
      else:                                                                                                             # 如果不存在，则走到死胡同，立即结束本次游走
        break
    return path

def load_edgelist(file_, undirected = True):                                                                            # 这里就是"rating_train.dat"这个数据集
  G = Graph()
  with open(file_,encoding="UTF-8") as f:
    for l in f:
      x, y = l.strip().split()[:2]                                                                                      # strip() 方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列，[:2]表示切片操作，即
      G[x].append(y)                                                                                                    # 即用户与哪些物品有关联，全部存在相应的key下。这种用.append()的添加方式就可以做到（这里貌似视为同构图奥，但根据编号能识别出是user还是item，因为它们分别有前缀u和i）
      if undirected:
        G[y].append(x)
  G.make_consistent()                                                                                                   # 其实就是去自环操作(一定要有这个操作！否则后面walk时会很慢！！！）
  return G

# 下面就是论文中的精髓的地方了，决定从每个root引出的游走路径的数目，以及在游走过程中的规则
def build_deepwalk_corpus_random(G, hits_dict, percentage, maxT, minT, alpha = 0, rand = random.Random()):
  walks = []
  nodes = list(G.nodes())                                                                                               # 返回由图G中每个节点的编号所组成的list，因为G继承了gul，所以这里可以像操作networkx一样操作它
  for node in nodes:
    num_paths = max(int(math.ceil(maxT * hits_dict[node])),minT)                                                        # 确定以每个node为root将要进行的随机游走的次数（即产生的路径的条数），用到了之前计算出的中心性指标的数值，这是论文上计算的方法
    for cnt in range(num_paths):                                                                                        # 这也是论文中的算法
      walks.append(G.random_walk_restart(nodes, percentage, rand = rand, alpha = alpha, start = node))

  random.shuffle(walks)                                                                                                 # .shuffle()函数用于对序列中的元素进行重新排序
  return walks

if __name__ == '__main__':
    G = Graph()