# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
新手牢记步骤
1.数据预处理
2.模型的设计
3.模型的训练
4.模型的输出
5.训练好的模型输出（训练好，指得到了对应理想的权重）

 任务目标:
    随机创建一个单位长度为5的向量
    任务1 如果向量中所有奇数位置的元素之和大于所有偶数位置元素之和，则将该向量标记为正样本（1），反之则标记为负样本（0）。
    
    任务2 如果向量中所有质数位置的元素之和（位置1, 2, 3, 5, 7）大于所有非质数位置元素的平均值乘以向量长度（即10），则将该向量标记为正样本（1）；否则，标记为负样本（0）
"""
# 1.数据预处理
def build_sample():
    """
    随机创建一个单位长度为5的向量,并且在创建成功的时候，判断第一位是否大于第五位，，要是大于第五为，则返回当前x，跟1，否则返回x，0
    :return: 返回生产的x值，标签值
    """
    # 任务一
    # x = np.random.random(5)
    # if x[0] + x[2] + x[4] > x[1] + x[3]:
    #     return x, 1
    # else:
    #     return x, 0

    # 任务二
    # 定义向量的大小和元素的范围
    vector_size = 10
    range_min = -50
    range_max = 50

    # 生成随机向量
    x = np.random.uniform(range_min, range_max, vector_size)

    # 质数位置平均值
    prime_number_sum = x[0] + x[1] + x[2] + x[4] + x[6]
    # 非质数
    no_prime_number_sum = x[3] + x[5] + x[7] + x[8] + x[9]
    no_prime_number_avg = no_prime_number_sum/5
    if prime_number_sum > no_prime_number_avg*10:
        return x, 1
    else:
        return x, 0

# 1.1随机生产一批样本数据
# 正负样本均匀生成,
def build_dataset(total_sample_num):
    """
    创建一个循环，指定输入的的数字，来生产一个样本数量
    :param total_sample_num: 样本个数
    :return:
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        # 引用build_sample创建样本
        x,y = build_sample()
        X.append(x)
        Y.append([y])
        """
        从一个包含多个numpy.ndarray的列表创建PyTorch张量（tensor）的效率非常低。这是因为PyTorch需要逐个处理列表中的每个数组，
        这在数组数量较大时会导致显著的性能开销。
        为了避免这种低效，PyTorch建议在创建张量之前，先使用numpy.array()函数将列表转换为一个单一的NumPy数组。这样可以显著提高创建张量的效率
        ，因为你将一次性转换整个数组，而不是逐个元素。
        """
        # print(X)
        # print(Y)
    # 记得，返回的时候，都是要用float处理一下
    # X和Y转换为NumPy数组
    X = np.array(X,dtype=np.float32)
    Y = np.array(Y,dtype=np.float32)
    # return torch.FloatTensor(X), torch.LongTensor(Y).我这里，因为用了NumPy 数组如果还用这个转换为，则转换为长整型
    return torch.from_numpy(X), torch.from_numpy(Y)
#上面就是我们的数据，接下来要创建我们的模型设计，权重，权重初始化，学习率


# 2.模型的设计
# 定义一个类，来作为权重初始化，这里用线性层，用两层
class Torchmodel(nn.Module):
    def __init__(self,input_size):
        """
        初始化定义性层的个数，定义线性层的大小
        定义sigmoid函数
        定义损失函数
        :param input_size:指定输入的线性张量大小
        """
        super(Torchmodel, self).__init__()
        # 定义一层线性层，输出大小为1 ，因为Y的值是单向量，这里是定义线性层的大小，
        # 注意，这里传入，input_size的时候，其实权重已经随机初始化，
        self.linear = nn.Linear(input_size, 1)
        # 初始化中创建sigmoid，作用，
        self.activation = torch.sigmoid
        # 损失函数使用交叉熵
        """
        对于此次的机器学习任务，这里使用交叉熵跟均方差，他们的模型预测相差有点大，交叉熵曲线正确跟错误是一上一下交叉起来
        可能原因： 
        交叉熵损失函数鼓励模型输出接近0或1的明确分类结果，以匹配二分类任务的标签。这意味着模型在训练过程中，会尝试调整其参数，使输出的概率值远离决策边界（即0.5）
        ，以减少损失。然而，在实际应用中，特别是在数据集复杂或模型参数接近最优解时，模型的输出可能仍然在决策边界附近波动。
        
        这样定义了一次五层线性层层数，这里的交叉熵曲线还是没变动
        
        对于任务一，或者任务2，这里采样交叉熵函数都是如此，变动较大
        """
        # self.loss =nn.functional.cross_entropy
        self.loss =nn.functional.mse_loss #均方差这里的图像比较好

    # 创建方法
    def forward(self, x,y = None):
        """
        函数主要是处理权重的初始化，传入一个样本，然后对样本进行权重的初始化，
        由初始化权重来算预测的y
        :param x: 传入的样本x
        :param y: 真实值
        :return: 目的是返回预测值
        """
        # 上述初始化的时候，已经初始化权重了，这里调用linear的forward方法，之间映射出y的预测值
        x = self.linear.forward(x)
        # print(x)
        # print(self.linear.state_dict())
        # 拿到权重之后，这里要传入权重值，来对数据进行sigmoid
        """
            一个单一的输出值，用sigmoid有什么用？这里注意，这里的x = self.linear.forward(x)已经是输出了预测值了
            但是，这里没有经过激活函数的处理，来引入对于非线性的处理，这样模型才能学习到负责的函数映射，我们这个也算是一个分类处理
            所以这里对预测值进行sigmoid处理
            即使模型输出是一个标量值，使用Sigmoid函数仍然是有用的，因为它提供了一个概率解释，并且使得模型的输出可以直接用作逻辑决策的依据
            （例如，如果输出大于0.5，则预测为正类；如果小于或等于0.5，则预测为负类）
            。此外，Sigmoid函数的导数可以用来计算损失函数相对于模型权重的梯度，这对于反向传播和模型训练至关重要。
        """
        y_pred = self.activation(x)
        # 如果这里的y不是none值，说明传入了真实的y值，这里就返回真实值与预测值的损失函数值
        if y is not None:
            # 这里返回的时候，y最好要定义是浮点类型
            return self.loss(y_pred, y).to(torch.float32)
        else:
            return y_pred.to(torch.float32)


# 查看预测的y
# print(build_dataset(1))
# torch_model = Torchmodel(5)
# train_x, train_y = build_dataset(1)
# print(torch_model.forward(train_x))
# print(build_dataset(3))


# 3.模型的训练、
# 3.1创建一个测试函数,用来测试每轮模型的准确率
def evaluate(model):
    """
   测试每轮模型的准确率
    :param model: 传入一个模型
    :return: 返回测试结果
    """
    # 定义模型为训练模式，用eval()方法
    model.eval()
    # 定义训练的样本个数
    test_sample_num = 100
    # 调用函数创建数据x，y
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # 初始化正确个数与错误个数
    correct, wrong = 0, 0
    # 调用我们自定义的模型，传入x 得到y的预测值
    with torch.no_grad():#这里禁用梯度计算  ptorch会默认进行梯度追踪
        y_pred = model(x)
        for y_p,y_t in zip(y_pred,y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct, wrong

    """
    这里要注意，我们始终没有进行模型的训练，权重都是初始化随机的，
    """
# 3.2模型的训练
def main():
    # 配置好初始参数
    epoch_num = 20  #训练的轮数
    batch_size = 40  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 1.创建数据集
    train_x, train_y = build_dataset(train_sample)
    # 2.建立模型：
    model = Torchmodel(input_size)     #注意这里，输入5之后，有了初始化权重
    # 选择优化器
    log = []
    # optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    """
        这里可以进行调优 使用AdamW优化器来实现  动态调整学习率
        可以加速收敛并防止训练过程中的震荡。
    """
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epoch_num):

        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size): #5000个样本总数，每次训练20个，共5000/20 250小轮

            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # x = train_x.to(torch.float32)
            # y = train_y.to(torch.float32)
            # 损失函数的计算
            loss = model.forward(x, y)  # 计算loss
            """
            优化损失函数
            正则化：添加正则化项以防止过拟合
            通过在损失函数中加入正则化项来实现
            """
            # L2正则化项
            l2_reg = torch.tensor(0., device=x.device)
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)

            # 正则化系数
            lambda_l2 = 0.01

            # 添加L2正则化项到损失
            loss = loss + lambda_l2 * l2_reg

            # 内置函数计算梯度
            loss.backward()  # 计算梯度 --继承了nn..model 计算梯度后，这个计算后的值存储在.grad属性中，
            # print(td)
            optim.step()  # 更新权重 Adam中有step函数进行更新权重 ，一开始初始化了权重，自动调用.grad属性来获取梯度进行权重的更新
            optim.zero_grad()  # 梯度归零  optim中有 zero_grad函数还进行梯度归零
            watch_loss.append(loss.item())  #记录每轮20次样本的损失函数，这里小轮循环完之后，得到250个损失函数
            """
            loss.item() 是从PyTorch的 Tensor 对象中提取一个Python数值（标量）的方法。loss 变量通常是一个标量张量，
            表示一个训练批次的损失值。loss.item() 将这个张量转换为一个Python的数值类型，如 float
            """
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss))) #平均损失函数
        acc = evaluate(model)  # 测试本轮模型结果 有准确个数，正常个数
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型  训练完之后，保存得到的权重值跟偏置值
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 4.模型的输出
# 使用训练好的模型做预测
def predict(model_path, input_vec):
    """

    :param model_path: 模型测试后的权重
    :param input_vec: 输出测试的样本数据
    :return: 拿了测试模型的权重之后，算权重之后，返回这次模型的预测结果
    """
    input_size = 5
    model = Torchmodel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 定义模型为训练模式，用eval()方法
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print(result)
    for vec, res in zip(input_vec, result):
        # round(float(res))对 输入正确权重后，得到的概率值进行四舍五入，然后输出模型的预测分类
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

if __name__ == "__main__":
    # 输出模型的测试结果
    main()
    # 5.训练好的模型输出
    # test_vec = [[0.8785, 0.8273, 0.1709, 0.3431, 0.9105],
    #     [0.4435, 0.1258, 0.4359, 0.6928, 0.0908],
    #     [0.0543, 0.6143, 0.0870, 0.1697, 0.5606],
    #     [0.6092, 0.1571, 0.0744, 0.6650, 0.9695],
    #     [0.6560, 0.8661, 0.1784, 0.3148, 0.2340]]
    # predict("model.pt", test_vec)

# 使用训练好的模型做预测










