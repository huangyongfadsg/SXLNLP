import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import json

"""
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符你所在位置进行分类
对比rnn和pooling做法
"""

#1.数据预处理
def build_vocab():
    chars = "你abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

# 1.1随机生成一个样本
# 传入字符集，传入想要的字符长度（或者叫字符个数）
def build_sample(vocab,sentence_length):
#随机从字典表中选择sentence_length个字，可能会重复,并且保证 你 在里面
#self.pool层的使用：你使用了nn.AvgPool1d(sentence_length)，这会将输入的最后一个维度（sentence_length）池化到1，因此输出的维度会变成(batch_size, vector_dim, 1)。然后你使用squeeze()将这个维度去掉，得到(batch_size, vector_dim)。
#self.classify层的输出：你的self.classify是一个nn.Linear(vector_dim, 6)，这意味着它会将(batch_size, vector_dim)的输入转换成(batch_size, 6)的输出，这是正确的，因为它应该输出6个位置的概率。
#目标y的维度：然而，你的目标y的维度是(batch_size, 1, sentence_length)，这是因为你在build_dataset函数中，将每个y样本放入一个列表中，然后将这个列表放入dataset_y中，最后将dataset_y转换成torch.FloatTensor，这导致每个样本的y标签变成了一个形状为(1, sentence_length)的张量，然后在build_dataset函数的最后，将所有样本的y标签堆叠在一起，得到一个形状为(batch_size, 1, sentence_length)的张量。
#为了解决这个问题，你需要确保你的目标y的维度匹配模型的输出维度。在你的案例中，你的模型输出是(batch_size, 6)，因此你的目标y也应该是一个形状为(batch_size, 6)的张量，其中每一行是一个one-hot编码向量，表示“你”字符的位置。这意味着在build_dataset函数中，你需要修改y标签的生成方式，确保每个y标签是一个one-hot编码向量，表示“你”字符在句子中的位置，而不是一个二进制向量。
    x = ['你']
    x += random.sample([char for char in vocab.keys() if char != '你'], sentence_length-1)
    # 打乱列表
    random.shuffle(x)
    # 找到“你”的位置
    you_position = x.index('你')
    # 创建one-hot编码的y标签
    y = [0.0] * sentence_length
    y[you_position] = 1.0
    # print(x)
    # 判断什么条件为正样本
    x = [vocab.get(word,vocab['unk']) for word in x]
    # 这里返回的是一个列表返回的是[0,0,0,1,0],代表第四类
    return  x,y

# 1.1建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length,vocab,sentence_length):
    """
    :param sample_length: 样本数量
    :param vocab: 字符集
    :param sentence_length: 字符长度
    :return: 将样本转化后，返回他的样本张量
    """
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y) # 直接添加y，不需要嵌套列表
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 2.模型的设计（torch模型）
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab,hidden_size = 30):
        """

        :param vector_dim: 传入字符的向量长度，单个字符
        :param sentence_length:传入字符的个数
        :param vocab: 字符集
        :return:
        """
        super(TorchModel,self).__init__()
        # 创建一个embedding层。这里是为了处理文本映射成向量
        self.embedding = nn.Embedding(len(vocab),vector_dim)

        self.RNN = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)

        # 创建一个池化层pooling层，来增加模型的健壮性，注意，pooling的输出维度是
        # 输出进来的字符个数长度除以传进来的参数  比如不给sentence_length传入pooling。传入的是2
        # 则输出的向量维度是sentence_length/2 = 维度。 但是这里我传入的是sentence_length，所以输出的维度是1维
        # self.pool = nn.AvgPool1d(sentence_length)

        # 创建线性层，选择了y = wx +b 的函数
        # 指定输出进来的向量长度，指定输出的向量长度，这里指定输出6维，因为我们是6个字符长度，是六分类任务
        self.classify = nn.Linear(hidden_size,6)
        # 创建sigmoid归一函数,sigmoid适合二元分类
        # self.activation = torch.sigmoid
        # self.activation = torch.softmax
        # loss函数采用均方差的方式
        # self.loss = nn.functional.mse_loss
        #交叉熵函数，传入真实[0,1,0]  ，预测[0.2,0.6,0.2]
        """
        self.BCELoss 没有进行softmax。 CrossEntropyLoss默认进行softmax
        确保在TorchModel类的forward方法中，你返回的y_perd和你传入的y在形状上是兼容的，特别是当你使用nn.CrossEntropyLoss()时，y应该是一个形状为(batch_size,)的Tensor，表示每个样本的类标签索引，
        而y_perd应该是一个形状为(batch_size, num_classes)的Tensor，表示每个类别的未归一化得分。
        
        BCEWithLogitsLoss这里要是分注意传参传的是，样本正确标签是第一类，则，y_true=[1,0,0],模型预测y_pred[0.5,0.4,0.1]  课本上的公式，进行了sigmoid
        
        CrossEntropyLoss 传参 模型预测y_pred = [0.5,0.4,0.1]，y_true = [1]两个有所区别，进行了sigmoid
        """
        self.loss = nn.CrossEntropyLoss()


    def forward(self,x,y=None):
        # 传入的x值经过embedding层，转化为向量
        x = self.embedding(x)          #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # 这里使用RNN注意：因为循环神经网络在处理序列数据时，不仅产生一个输出序列，还产生一个最终的隐藏状态，这个状态可以被用作序列级别的特征，或者用于初始化下一个序列的隐藏状态
        # x= torch.LongTensor(x)
        output, _ = self.RNN(x)
        output = output[:, -1, :]
        # 转化为embedding之后，传入pooling层，我们要先转置，不然默认对最后一维进行pooling
        # x = x.transpose(1,2)            #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)            #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # 因为pooling出来的结果就想转置一样，是5*1的，我们其实要的是1*5的
        # x = x.squeeze()             #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # 转置之后，得到一个pooling的平均值结果的向量，将这个向量传入我们的全连接层
        # 这里就得到一个个预测的y值
        # output = torch.FloatTensor(output)
        y_perd = self.classify(output)
        # y_perd = self.activation(y_perd, dim=1)  # 应用softmax函数，并指定沿着类别维度操作  手动调用softmax
        if y is not None:
            # 将 y_true 从 one-hot 编码转换为类别索引
            y= torch.argmax(y, dim=1)
            return self.loss(y_perd,y)
        else:
            return y_perd
# 2.1建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 3.模型的训练
# 3.1模型的测试,#用来测试每轮模型的准确率
def evaluate(model,vocab,sample_length):
    model.eval() #开启训练模式
    x,y = build_dataset(200,vocab,sample_length) #建立200个用于测试的样本
    if not isinstance(y, torch.Tensor):  # 检查y是否是Tensor
        y = torch.tensor(y, dtype=torch.float32)  # 如果不是，将其转换为Tensor
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct= 0
    with torch.no_grad():
        # 将测试样本x传入模型中，得到预测值y
        y_pred = model(x)   #模型预测
        y_pred_class = torch.argmax(y_pred, dim=1)  # 获取预测类别
        y_true_class = torch.argmax(y, dim=1)  # 获取真实类别
        correct = (y_pred_class == y_true_class).sum().item()
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/200))
    return correct/200

def main():
    #配置参数
    epoch_num =20           #训练的轮数
    batch_size = 20         #每轮训练的样本数量
    train_sample = 500      #每轮训练总共训练的样本总数
    char_dim = 20           #每个字的维度
    sentence_length = 6     #样本文本长度
    learning_rate = 0.005  # 学习率

    # 1.建立数据（建立字段表）
    vocab = build_vocab()
    # 2.建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 2.1.选择优化器
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model.forward(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path,vocab_path,input_strings):
    char_dim = 20 # 每个字的维度
    sentence_length = 6 # 样本文本长度
    vocab = json.load(open(vocab_path,'r',encoding= 'utf8'))
    # 建立模型
    model = build_model(vocab,char_dim,sentence_length)
    model.load_state_dict(torch.load(model_path)) #加载训练好的权重
    x= []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string]) #将输入序列化
    model.eval() #测试模式
    with torch.no_grad(): #不计算梯度
        result = model.forward(torch.LongTensor(x)) #模型预测
    print(result)
    # 假设result是模型的输出，形状为(batch_size, num_classes)

    # 使用argmax找到预测类别
    predicted_classes = torch.argmax(result, dim=1)
    print(predicted_classes)
    # 计算每个预测类别的概率
    probabilities = torch.gather(torch.softmax(result, dim=1), 1, predicted_classes.unsqueeze(1)).squeeze()

    for i, input_string in enumerate(input_strings):
        print(
            "输入：%s, 预测类别：%d, 概率值：%.4f" % (input_string, predicted_classes[i].item(), float(probabilities[i])))

if __name__ == "__main__":
    main()
    test_strings = ["f你vfee", "wz你dfg", "rqadeg", "naawww"]
    predict("model.pth", "vocab.json", test_strings)