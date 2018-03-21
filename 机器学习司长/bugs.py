# coding=utf-8
from math import log
# @dataSet 传入的数据集 原始数据
#     [
#         [1,1,'yes'],
#         [2,3,'no'],
#         [1,2,'yes'],
#         [2,3,'no'],
#     ]

def cal_shonnan_entropy(dataSet):
    lens = len(dataSet)
    s = {}
    for items in dataSet:
        feature = items[-1]
        print("feature",feature);
        if feature not in s.keys():
            s[feature] = 0
        s[feature]+=1
    entropy = 0
    for items in s:
        pro = s[items]/lens
        entropy += pro*log(pro,2)
    return entropy

# def listToStr(listt):
#     strf = ''
#     for i in listt:
#         print(i)
#         if(isinstance(i,list)):
#             strf += listToStr(i[1])
#             return strf
#         else:
#             strf += str(i)



# t = [
#      [1,1,'yes',1.03,[1,[2,3],2,3]],
#      [1,1,'yes'],
#      [1,0,'no'],
#      [0,1,'no'],
#      [1,0,'no'],
#     ]

# print(cal_shonnan_entropy(t))
# print(t[0])
# print(isinstance(t[0],list))
# print(listToStr(t[0]))
'''
 按照给定的数据划分数据集
   dataSet =  [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'] ; [0, 1,'no']]
   axis = 0/1
   value = dataSet[0:-1][0:-1]
'''

# def splitDataSet(dataSet,axis,value):
#     retData = []
#     for data in dataSet:
#         if data[axis] == value:
#             reduce_data = data[:axis]
#             reduce_data.extend(data[axis+1:])
#             retData.append(reduce_data)
#     return retData
#
#
# dataSet =  [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],[0, 1,'no']]
# print( splitDataSet(dataSet,0,1))
# print( splitDataSet(dataSet,0,0))
'''
    输入: 训练集D = {(x1,y1),(x2,y2),...(xm,ym)}
            属性集 A = {a1,a2,a3,,,ad}
    1.生产一个节点node
    2. if D 中的样本属于同一类别C
         将node标记为C类叶子节点; return
       end if
    3  if A = 空 或者 D中样本在A的取值都一样
          将node 标记为叶子节点,类别是D中样本最多的类,
    4  从A中选择最佳的划分a

'''
'''
    返回label最多的class
'''



def dict_to_list(dic):
    keys = dic.keys()
    vals = dic.values()
    lst = [(key,val) for key,val in zip(keys,vals)]
    return lst


def majority_choose(class_list):
    val = {}
    for items in class_list:
        if items not in val.keys():
            val[items] = 1
        else:
            val[items] += 1
    print('val', val)
    lst = dict_to_list(val)
    print(lst)
    sorted_dict = sorted(lst, key=lambda x: x[1],reverse=True)[0] # 按照数字求最大的 ascending
    for items in sorted_dict:
        if isinstance(items, str):
            return items
        return ''

def split_data_set(dataSet,index,value):
    ret_data_set = {}
    for item in dataSet:
        if item[index] == value:
            reduce_feature = dataSet[:index]
            reduce_feature.extend(dataSet[index+1:-1])
        ret_data_set.append(reduce_feature)
    return ret_data_set




def choose_best_feature_to_split(dataSet):
    number_features = len(dataSet[0]) - 1 #最后的一个是label
    base_entropy = cal_shonnan_entropy(dataSet)
    max_info_diff = 0.0
    best_index = -1
    for i in range(number_features):
        feature_list = [example[i] for example in dataSet] # 取到每一个feature
        unique = set(feature_list)
        for unquireval in unique:
            sub_class = split_data_set(dataSet,i,unquireval)
            cal_info_diff = base_entropy - cal_shonnan_entropy(sub_class)
            if cal_info_diff > max_info_diff:
                max_info_diff = cal_info_diff
                best_index = i
    return best_index






def create_tree(dataSet,labels):
    class_list = [ex[-1] for ex in dataSet] #list_comprehension
    if class_list.count(class_list[0]) == len(class_list):  #所有的样本属于同一个类型
        return class_list[0]
    if 1 == len(class_list):
        return majority_choose(class_list) #返回样本中最多的class
    best_choose = choose_best_feature_to_split(dataSet)
    best_feature_label = labels[best_choose]
    my_tree = {best_feature_label:{}}
    del(labels[best_choose])
    featValues = [ex[best_choose] for ex in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        my_tree[best_feature_label][value] = create_tree(split_data_set(
            dataSet,best_choose,value),subLabels)
    return my_tree


#
# dict = {
#     'yes':1,
#     'no':2,
#     'no':2,
# }

# print(dict_to_list(dict))

dic = {'a':3 , 'b':2 , 'c': 1}
# print(sorted(dict_to_list(dic), key=lambda x: x[0], reverse=True))


dataSet =  [
    [1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1,'no']
];


# print(create_tree(dataSet,None));



dataSet1 =  [
    'yes',
    'yes',
    'no',
    'no',
    'no'
];

print(majority_choose(dataSet1))

