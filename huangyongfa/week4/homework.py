#week3作业
import re
import time
import json

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"



#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, start=0, path=None, result=None):
    if path is None:
        path = []
    if result is None:
        result = []

    if start == len(sentence):
        # print('*****',start)
        result.append(path[:])
        return

    for end in range(start + 1, len(sentence) + 1):
        # print(start,end)
        #一层循环 0,1
        word = sentence[start:end]
        if word in Dict:
            #word = 经
            path.append(word)
            #path = [经]
            all_cut(sentence, Dict, end, path, result)
            # 当if条件不满足的时候,回溯递归,撤回上一个加的字
            path.pop()  # backtracking
    # end  = 1 传入start
    # for end in range(start + 1, len(sentence) + 1):
    #     # print(start,end)
    #     #一层循环 1,2
    #     word = sentence[start:end]
    #     if word in Dict:
    #         #word = 常
    #         path.append(word)
    #         #path = [经,常]
    #         all_cut(sentence, Dict, end, path, result)
    #         path.pop()  # backtracking

    # end  = 6 传入start
    # for end in range(start + 1, len(sentence) + 1):
    #     # print(start,end)
    #     #循环 6,7
    #     word = sentence[start:end]
    #     if word in Dict:
    #         #word = 歧
    #         path.append(word)
    #         #path = [经,常,有,意,见,分,歧]
    #         all_cut(sentence, Dict, end, path, result)
    #         path.pop()  # backtracking

    # 末层循环
    # end  = 7 传入start
    # 此时start == len(sentence):满足,则result.append(path[:])
    # result.append(path[:]) ==>[经,常,有,意,见,分,歧]
    # for end in range(start + 1, len(sentence) + 1):
    #     # print(start,end)
    #     #循环 5,7
    #     word = sentence[start:end]
    #     此时,没有word满足条件.这里时候,回溯递归, [经,常,有,意,见,分]
    #     在时候,切分start = 5 ,end = 7,直到切分word 存在,
    #     if word in Dict:
    #         #word = [分歧]
    #         path.append(word)
    #         #path = [经,常,有,意,见,分,歧，‘分歧’]  ==》最终这里会的分，歧会被替换为 ‘分歧’
    #         all_cut(sentence, Dict, end, path, result)
    #         path.pop()  # backtracking
    return result

target = all_cut(sentence, Dict)

# 打印结果
for t in target:
    print(t)



