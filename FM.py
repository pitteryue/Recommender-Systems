import numpy as np
import torch

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        return torch.sum(self.fc(x), dim=1) + self.bias

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        return self.embedding(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FactorizationMachineModel(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


#下面是主程序里面的，到时代码修复后可以直接放进main.py
    '''
    elif model_name == 'FM':
        user_one_hot, item_one_hot = data_ratings.ONE_HOT_ENCODING(train_dataset)
        user_dim = user_one_hot.columns.size                                                                            # 这里应该搞错了吧，或许应该用max_user+1才对，nn.Embedding()的编号
        item_dim = item_one_hot.columns.size
        field_dims = [user_dim, item_dim]

        fm = FactorizationMachineModel(field_dims, n_latent)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(fm.parameters(), lr=0.01, weight_decay=0.5)

        for epoch in range(max_epoch):
            train_loss = 0
            s = 0.
            for row in train_dataset.iloc():
                input = Variable(torch.from_numpy(np.array((pd.concat([user_one_hot.loc[row['user'], :],item_one_hot.loc[row['item'],:]]).values.tolist())))).unsqueeze(0)
                input = input.long()
                output = fm(input)
                target = Variable(torch.from_numpy(np.array((row['rating']).tolist()))).unsqueeze(0).float()
                loss = criterion(output, target)
                loss.backward()
                train_loss += loss.item()
                s += 1.
                optimizer.step()

            print('epoch:' + str(epoch) + 'loss' + str(train_loss / s))
    '''