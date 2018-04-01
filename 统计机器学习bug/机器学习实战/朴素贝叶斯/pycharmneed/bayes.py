from numpy import *
def load_data_set():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vector = [0,1,0,1,0,1]
    return  postingList,class_vector


'''
    将dataSet的每一条词条去合并成一个词条返回出来
'''
def create_vacab_list(dataSet):
    vacab_set = set([])
    for document in dataSet:
        vacab_set = vacab_set | set(document)
    vab_list = list(vacab_set)
    vab_list.sort(reverse=True)
    # print(vab_list)
    return list(vab_list)


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

# for item in posting_list:
#     print(set_of_words2_vec( vacab_list,item))


'''
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0] 
[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]

'''
def train_navie_bayes(train_matrix,train_class):
    nums_train_docs = len(train_matrix) # 获取文档的数目

    nums_words = len(train_matrix[0]) # 每一行的单词数目

    p_class =  sum(train_class)/nums_train_docs

    p0_nums=ones(nums_words);
    p0_sum=2
    p1_nums=ones(nums_words);
    p1_sum=2


    for i in range(nums_train_docs):
        if train_class[i] == 1:
            p1_nums += train_matrix[i]
            p1_sum += sum(train_matrix[i])
        else:
            p0_nums += train_matrix[i]
            p0_sum += sum(train_matrix[i])
    p1_vec = log(p1_nums/p1_sum)
    p0_vec = log(p0_nums/p0_sum)

    return p0_vec,p1_vec,p_class

train_matrix = [
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 0
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   #1
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],   #0
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],   #1
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],   #0
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],   #1
]

train_class = [0,1,0,1,0,1]


# print(train_navie_bayes(train_matrix,train_class))

'''
    
'''

def classify_bayes(vec2_classify,p0_vec,p1_vec,p_class):
    # print("vec2_classify",vec2_classify)
    p1 = sum(vec2_classify*p1_vec) + log(p_class) #实际是两个概率取乘积 条件概率相乘
    p0 = sum(vec2_classify*p0_vec) + log(1-p_class)
    if p1>p0:
        return 1
    else:
        return 0

def test_bayes_classify():
    list_posts,list_class = load_data_set()
    my_vocal_list = create_vacab_list(list_posts)
    train_mat = []

    for post_item in list_posts:
        train_mat.append(set_of_words2_vec(my_vocal_list,post_item))

    p0_vec,p1_vec,pA = train_navie_bayes(array(train_mat),array(list_class))

    test_entry_1 = ['love','my','dalmation']

    thisDoc1 = array(set_of_words2_vec(my_vocal_list,test_entry_1))

    print("1",classify_bayes(thisDoc1,p0_vec,p1_vec,pA))

    test_entry_2 = ['stupid','garbage']

    thisDoc2 = array(set_of_words2_vec(my_vocal_list, test_entry_2))

    print("0",classify_bayes(thisDoc2, p0_vec, p1_vec, pA))

# test_bayes_classify()



'''
    对词进行频次统计 
'''
def bag_of_words_2_vec_mn(vocab_list,input_set):
    return_vec = [0]*len(vacab_list)
    for word in input_set:
        if word in vacab_list:
            return_vec[vacab_list.index(word)] +=1
    return return_vec


def text_parse(text):
    import re
    list_tokens = re.split(r'\W*',text) 
    return [tok.lower() for tok in list_tokens if len(tok)>2]

# text = r'Never lie to someone who trust you, Never trust someone who lies to you.'
# print(text_parse(text))

email_path = 'F:\\BaiduNetdiskDownload\\机器学习\\机器学习实战\\machinelearninginaction\\machinelearninginaction\\Ch04\\email'

def spam_test():

    doc_list = [];
    class_list = [];
    full_text =[];

    ham_path = email_path + '\\ham\\';


    spam_path = email_path + '\\spam\\';


    for i in range(1,26):
        word_list = text_parse(open(ham_path+ '%d.txt'%i,encoding='utf-8').read())
        doc_list.append(word_list)
        class_list.append(1)
        full_text.append(word_list)

        word_list = text_parse(open(spam_path + '%d.txt'%i,encoding='utf-8').read())
        doc_list.append(word_list)
        class_list.append(0)
        full_text.append(word_list)

    vacab_list = create_vacab_list(doc_list)
    train_set = list(range(50))
    test_set = []

    for i in range(10):   #构建随机的训练集
        rand_index = int(random.uniform(0,len(train_set)))
        test_set.append(train_set[rand_index])
        del(train_set[rand_index])

    train_mat = []; train_class = []

    for doc_index in train_set:
        train_mat.append(set_of_words2_vec(vacab_list,doc_list[doc_index]))
        train_class.append(class_list[doc_index])

    p0_vec,p1_vec,p_spam = train_navie_bayes(array(train_mat),array(train_class))


    error_count = 0


    for doc_index in test_set:
        word_vector = set_of_words2_vec(vacab_list,doc_list[doc_index])
        if classify_bayes(array(word_vector),p0_vec,p1_vec,p_spam) != class_list[doc_index]:
            error_count +=1
    print('the error rate is ',float(error_count)/len(test_set))

# spam_test()


test_set_path = r'F:\BaiduNetdiskDownload\机器学习\机器学习实战\machinelearninginaction\machinelearninginaction\Ch05\testSet.txt'

def load_dataset():
    data_mat = []
    label_mat =[]

    file_open = open(test_set_path)
    for line in file_open.readlines():
        line_data = line.strip().split('\t')

        data_mat.append([float(line_data[0]),float(line_data[0])])

        label_mat.append([int(line_data[2])])

    return data_mat,label_mat

print(load_dataset())
































