import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):

	# 定义参数和搭建网络
	def __init__(self, user_num, item_num, n_latent, model, GMF_model = None, MLP_model=None, num_layers = 3, dropout=0.5):# 这后面的两个None是要直接读入模型的
		super(NCF, self).__init__()

		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model																						# 这俩是什么？？？
		self.MLP_model = MLP_model

		# 定义GMF和MLP的输入embeddings
		self.embed_user_GMF = nn.Embedding(user_num, n_latent)															# 这里暗示着输入数据的id必须得是连续的吗？？？
		self.embed_item_GMF = nn.Embedding(item_num, n_latent)
		self.embed_user_MLP = nn.Embedding(user_num, n_latent * (2 ** (num_layers - 1)))								# 相当MLP训练时经过每一层后embedding的dim减少一半
		self.embed_item_MLP = nn.Embedding(item_num, n_latent * (2 ** (num_layers - 1)))

		# 定义MLP深度网络
		MLP_modules = []																								# 定义一个list
		for i in range(num_layers):																						# 定义DNN的办法
			input_size = n_latent * (2 ** (num_layers - i))
			MLP_modules.append(nn.Dropout(p = self.dropout))															# Dropout被设计为只在训练中使用，当需要对模型进行预测或评估时，则需要关闭，很方便，进入eval()就自动关闭了
			MLP_modules.append(nn.Linear(input_size, input_size//2))													# //是求整商，而/是真除法
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)																	#nn.Sequential()用来连接多个层，这里为什么要用指针呢？

		# 定义输出层
		if self.model in ['MLP', 'GMF']:
			predict_size = n_latent
		else:
			predict_size = n_latent * 2

		# 定义预测操作
		self.predict_layer = nn.Linear(predict_size, 1)

		# 初始化参数
		self._init_weight_()																							# 定义好参数和网络结构后，在__init__中的最后一步就是初始化输入和参数

 	# 初始化参数
	def _init_weight_(self):
		if not self.model == 'NeuMF':

			# 初始化embeddings输入
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)														# 初始化embeddings的操作（即对应的连边权重组成的向量）
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			# 初始化MLP层的权重
			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):																			# isinstance(object,classinfo)，如果object的类型与classinfo的类型相同，则返回True，否则返回False
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')							# xavier_uniform和kaiming_uniform都是不同的初始化均匀分布策略，前者针对sigmoid和tanh，后者针对ReLU

			# 初始化所有线性层的偏置值
			for m in self.modules():																					# .modules返回一个包含当前模型所有模块的迭代器，且采用深度优先搜索策略
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers，把分别训练得到的GMF和MLP的embeddings作为NCF的初始化参数
			self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

			# MLP layers，同样把MLP的权重参数赋予NCF作为其初始化
			for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):											# zip(obj1,obj2)函数将对象中位置对应的元素打包成一个个元组(,)
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([self.GMF_model.predict_layer.weight,self.MLP_model.predict_layer.weight], dim=1)# cat(A,B,dim)函数，将两个张量A和B拼接在一起，dim=0是竖着拼，dim=1是横着拼
			precit_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	# 向前传播
	def forward(self, user, item):

		if not self.model == 'MLP':																						# 两个if not，就是说对于NeuMF,都要执行一遍！
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF																# 这里对张量的*运算符就是element-wise，注意不是点积哦，点积有专门的dot()函数

		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)

		return prediction.view(-1)																						# .view的作用是reshape the tensor, 参数-1表示将多行的tensor拼接成一行