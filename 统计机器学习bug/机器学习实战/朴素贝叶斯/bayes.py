def load_data_set():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vector = [0,1,0,1,1]
    return  postingList,class_vector


'''
    将dataSet的每一条词条去合并成一个词条返回出来
'''
def create_vacab_list(dataSet):
    vacab_set = set([])
    for document in dataSet:
        vacab_set = vacab_set | set(document)
    return list(vacab_set)


posting_list,class_vector = load_data_set();
vacab_list = create_vacab_list(posting_list)
# print(vacab_list)


'''
    @:varb_list 输入的list 元素唯一的
    @:input_set 输入的一条文本
    @return 返回一个list [0 ,1] 1 表示input_set的词在vocab_list 
    变成了一个词向量 变成的feature
'''
def set_of_words2_vec(vocab_list,input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("the word %s is not in my vacabulary!",word)
    return  return_vec


print(set_of_words2_vec( vacab_list,posting_list[1]))


def train_navie_bayes(train_matrix,train_class):
    nums_train_docs = len(train_matrix) # 获取文档的数目

    nums_words = len(train_matrix[0]) # 每一行的单词数目

