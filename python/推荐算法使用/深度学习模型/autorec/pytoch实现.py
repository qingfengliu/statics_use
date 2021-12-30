import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

def dataProcess(filename, num_users, num_items, train_ratio):
    fp = open(filename, 'r')
    lines = fp.readlines()

    num_total_ratings = len(lines)

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    train_r = np.zeros((num_users, num_items))
    test_r = np.zeros((num_users, num_items))

    train_mask_r = np.zeros((num_users, num_items))
    test_mask_r = np.zeros((num_users, num_items))

    # 生成0~num_total_ratings范围内的的随机序列
    random_perm_idx = np.random.permutation(num_total_ratings)
    # 将数据分为训练集和测试集
    train_idx = random_perm_idx[0:int(num_total_ratings * train_ratio)]
    test_idx = random_perm_idx[int(num_total_ratings * train_ratio):]

    ''' Train '''
    for itr in train_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        train_r[user_idx][item_idx] = int(rating)
        train_mask_r[user_idx][item_idx] = 1

        user_train_set.add(user_idx)
        item_train_set.add(item_idx)

    ''' Test '''
    for itr in test_idx:
        line = lines[itr]
        user, item, rating, _ = line.split("::")
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        test_r[user_idx][item_idx] = int(rating)
        test_mask_r[user_idx][item_idx] = 1

        user_test_set.add(user_idx)
        item_test_set.add(item_idx)

    return train_r, train_mask_r, test_r, test_mask_r, user_train_set, item_train_set, user_test_set, item_test_set


def Construct_DataLoader(train_r, train_mask_r, batchsize):
    torch_dataset = Data.TensorDataset(torch.from_numpy(train_r), torch.from_numpy(train_mask_r))
    return Data.DataLoader(dataset=torch_dataset, batch_size=batchsize, shuffle=True)


autorec_config = \
{
    'train_ratio': 0.9,
    'num_epoch': 200,
    'batch_size': 100,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'l2_regularization':1e-4,
    'num_users': 6040,
    'num_items': 3952,
    'hidden_units': 500,
    'lambda': 1,
    'device_id': 2,
    'use_cuda': False,
    'data_file': 'D:/model/ml-1m/ratings.dat',
    'model_name': 'D:/model/ml-1m/AutoRec.model'
}

class AutoRec(nn.Module):
    """
    基于物品的AutoRec模型
    """
    def __init__(self, config):
        super(AutoRec, self).__init__()
        self._num_items = config['num_items']
        self._hidden_units = config['hidden_units']
        self._lambda_value = config['lambda']
        self._config = config

        # 定义编码器结构
        self._encoder = nn.Sequential(
            nn.Linear(self._num_items, self._hidden_units),
            nn.Sigmoid()
        )
        # 定义解码器结构
        self._decoder = nn.Sequential(
            nn.Linear(self._hidden_units, self._num_items)

            
        )

    def forward(self, input):
        return self._decoder(self._encoder(input))

    def loss(self, res, input, mask, optimizer):
        cost = 0
        temp = 0

        cost += ((res - input) * mask).pow(2).sum()
        rmse = cost

        for i in optimizer.param_groups:
            # 找到权重矩阵V和W，并且计算平方和，用于约束项。
            for j in i['params']:
                if j.data.dim() == 2:
                    temp += torch.t(j.data).pow(2).sum()

        cost += temp * self._config['lambda'] * 0.5
        return cost, rmse

    def recommend_user(self, r_u, N):
        """
        :param r_u: 单个用户对所有物品的评分向量
        :param N: 推荐的商品个数
        """
        # 得到用户对所有物品的评分
        predict = self.forward(torch.from_numpy(r_u).float())
        predict = predict.detach().numpy()
        indexs = np.argsort(-predict)[:N]
        return indexs

    def recommend_item(self, user, test_r, N):
        """
        :param r_u: 所有用户对物品i的评分向量
        :param N: 推荐的商品个数
        """
        # 保存给user的推荐列表
        recommends = np.array([])

        for i in range(test_r.shape[1]):
            predict = self.forward(test_r[:, i])
            recommends.append(predict[user])

        # 按照逆序对推荐列表排序，得到最大的N个值的索引
        indexs = np.argsot(-recommends)[:N]
        # 按照用户对物品i的评分降序排序吗，推荐前N个物品给到用户
        return recommends[indexs]

    def evaluate(self, test_r, test_mask_r, user_test_set, user_train_set, item_test_set, item_train_set):
        test_r_tensor = torch.from_numpy(test_r).type(torch.FloatTensor)
        test_mask_r_tensor = torch.from_numpy(test_mask_r).type(torch.FloatTensor)

        res = self.forward(test_r_tensor)

        unseen_user_test_list = list(user_test_set - user_train_set)
        unseen_item_test_list = list(item_test_set - item_train_set)

        for user in unseen_user_test_list:
            for item in unseen_item_test_list:
                if test_mask_r[user, item] == 1:
                    res[user, item] = 3

        mse = ((res - test_r_tensor) * test_mask_r_tensor).pow(2).sum()
        RMSE = mse.detach().cpu().numpy() / (test_mask_r == 1).sum()
        RMSE = np.sqrt(RMSE)
        print('test RMSE : ', RMSE)

    def saveModel(self):
        torch.save(self.state_dict(), self._config['model_name'])

    def loadModel(self, map_location):
        state_dict = torch.load(self._config['model_name'], map_location=map_location)
        self.load_state_dict(state_dict, strict=False)
def pick_optimizer(network, params):
    optimizer = None
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=params['adam_lr'],
                                     weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

class Trainer(object):
    def __init__(self, model, config):
        self._model = model
        self._config = config
        self._optimizer = pick_optimizer(self._model, self._config)

    def _train_single_batch(self, batch_x, batch_mask_x):
        """
        对单个小批量数据进行训练
        """
        if self._config['use_cuda'] is True:
            # 将这些数据由CPU迁移到GPU
            batch_x, batch_mask_x = batch_x.cuda(), batch_mask_x.cuda()

        # 模型的输入为用户评分向量或者物品评分向量，调用forward进行前向传播
        ratings_pred = self._model(batch_x.float())
        # 通过交叉熵损失函数来计算损失, ratings_pred.view(-1)代表将预测结果摊平，变成一维的结构。
        loss, rmse = self._model.loss(res=ratings_pred, input=batch_x, mask=batch_mask_x, optimizer=self._optimizer)
        # 先将梯度清零,如果不清零，那么这个梯度就和上一个mini-batch有关
        self._optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 梯度下降等优化器 更新参数
        self._optimizer.step()
        # 将loss的值提取成python的float类型
        loss = loss.item()
        return loss, rmse

    def _train_an_epoch(self, train_loader, epoch_id, train_mask):
        """
        训练一个Epoch，即将训练集中的所有样本全部都过一遍
        """
        # 告诉模型目前处于训练模式，启用dropout以及batch normalization
        self._model.train()
        total_loss = 0
        total_rmse = 0
        # 从DataLoader中获取小批量的id以及数据
        for batch_id, (batch_x, batch_mask_x) in enumerate(train_loader):
            assert isinstance(batch_x, torch.Tensor)
            assert isinstance(batch_mask_x, torch.Tensor)

            loss, rmse = self._train_single_batch(batch_x, batch_mask_x)
            # print('[Training Epoch: {}] Batch: {}, Loss: {}, RMSE: {}'.format(epoch_id, batch_id, loss, rmse))
            total_loss += loss
            total_rmse += rmse
        rmse = np.sqrt(total_rmse.detach().cpu().numpy() / (train_mask == 1).sum())
        print('Training Epoch: {}, Total Loss: {}, total RMSE: {}'.format(epoch_id, total_loss, rmse))

    def train(self, train_r, train_mask_r):
        # 是否使用GPU加速
        self.use_cuda()

        for epoch in range(self._config['num_epoch']):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            # 构造一个DataLoader
            data_loader = Construct_DataLoader(train_r, train_mask_r, batchsize=self._config['batch_size'])
            # 训练一个轮次
            self._train_an_epoch(data_loader, epoch_id=epoch, train_mask=train_mask_r)

    def use_cuda(self):
        if self._config['use_cuda'] is True:
            assert torch.cuda.is_available(), 'CUDA is not available'
            torch.cuda.set_device(self._config['device_id'])
            self._model.cuda()

    def save(self):
        self._model.saveModel()

if __name__ == "__main__":
    ####################################################################################
    # AutoRec 自编码器协同过滤算法
    ####################################################################################
    train_r, train_mask_r, test_r, test_mask_r, \
    user_train_set, item_train_set, user_test_set, item_test_set = \
        dataProcess(autorec_config['data_file'], autorec_config['num_users'], autorec_config['num_items'], autorec_config['train_ratio'])
    # 实例化AutoRec对象
    autorec = AutoRec(config=autorec_config)

    ####################################################################################
    # 模型训练阶段
    ####################################################################################
    # 实例化模型训练器
    trainer = Trainer(model=autorec, config=autorec_config)
    # 开始训练
    trainer.train(train_r, train_mask_r)
    # 保存模型
    trainer.save()

    ###################################################################################
    # 模型测试阶段
    ###################################################################################
    # autorec.loadModel(map_location=torch.device('cpu'))
    #
    # # 从测试集中随便抽取几个用户，推荐5个商品
    # print("用户1推荐列表： ",autorec.recommend_user(test_r[0], 5))
    # print("用户2推荐列表： ",autorec.recommend_user(test_r[9], 5))
    # print("用户3推荐列表： ",autorec.recommend_user(test_r[23], 5))
    #
    # autorec.evaluate(test_r, test_mask_r, user_test_set=user_test_set, user_train_set=user_train_set, \
    #                  item_test_set=item_test_set, item_train_set=item_train_set)
