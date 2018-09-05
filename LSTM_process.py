import multiprocessing
import os
import re
import codecs
import pandas as pd
import numpy as np
import jieba
import jieba.posseg
import yaml

from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

vocab_dim = 300#词向量维度
n_exposures = 10#10
window_size = 7
cpu_count = multiprocessing.cpu_count()
n_iterations = 1
max_len = 10
input_length = 100
batch_size = 16
n_epoch = 4
# 数据预处理操作：分词，去停用词，词性筛选
def dataPrepos(text, stopkey):
    l = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
    # print(text)
    seg = jieba.posseg.cut(text)  # 分词
    #seg = [jieba.lcut(document.replace('\n', '')) for document in text]
    # print('---'*45)
    #print(seg)
    for i in seg:
        if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
            l.append(i.word)
    #print(l)
    return l
#获取数据
def get_datanew():
    root_path = os.path.dirname(os.path.abspath(__file__))
    filepath1 = os.path.normpath(os.path.join(root_path, '../data/neg_test.txt'))
    filepath2 = os.path.normpath(os.path.join(root_path, '../data/pos_test.txt'))
    re_split = re.compile("\s+")
    # 停用词表

    stopkey = [w.strip() for w in codecs.open('../data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    # print(stopkey)
    pos_doc_list = []
    neg_doc_list = []
    # 将文本中的数据读入self函数中，并按照标签储存
    with open(filepath1, encoding='utf-8') as f:  # utf-8, encoding='gb18030', errors='ignore'
        # print(f)
        for line in f:
            # print(line)
            line = dataPrepos(line, stopkey)

            neg_doc_list.append(line[1:])

    with open(filepath2, encoding='utf-8') as f:
        for line in f:
            line = dataPrepos(line, stopkey)

            pos_doc_list.append(line[1:])

    pos_doc_length = len(pos_doc_list)
    neg_doc_length = len(neg_doc_list)
    combined = np.concatenate((pos_doc_list, neg_doc_list))
    y=[1] * (pos_doc_length) + [0] * (neg_doc_length)
    return combined, y
# 加载文件
def loadfile():
    neg = pd.read_excel('../data/neg.xls', sheet_name=0, header=None, index=None)
    pos = pd.read_excel('../data/pos.xls', sheet_name=0, header=None, index=None)

    combined = np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))

    return combined, y


# 对句子进行分词，并取掉换行
def tokenizer(text):

    '''Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,#300 特征向量的维度
                     min_count=n_exposures,#可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
                     window=window_size,#表示当前词与预测词在一个句子中的最大距离是多少
                     workers=cpu_count,#参数控制训练的并行数。
                     iter=n_iterations)# 迭代次数
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)#
    model.save('../lstm_data/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)

    return index_dict, word_vectors, combined


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    maxlen = 100
    ''' Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and model is not None:
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}  # 所有词频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()} # 所有词频数超过10的词语的词向量

        def parse_dataset(combined):
            '''
            Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined=sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provide')


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，词频小于10的词语索引为0，所以加1

    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    print(embedding_weights)
    print('888'*34)
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train)
    print('---'*45)
    print(x_test)
    print('---' * 45)
    print(y_train)
    print('---' * 45)
    print(y_test)
    print('---' * 45)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

    print('Defining a simple Keras Model')

# 定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    ## 定义基本的网络结构
    model = Sequential() # or Graph or whatever
    ## 对于LSTM 变长的文本使用Embedding 将其变成指定长度的向量
    model.add(Embedding(output_dim=vocab_dim,#大于0的整数，代表全连接嵌入的维度
                        input_dim=n_symbols,#大或等于0的整数，字典长度，即输入数据最大下标+1
                        mask_zero=True,#布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用递归层处理变长输入时有用。
                        weights=[embedding_weights],
                        input_length=input_length))#当输入序列的长度固定时，该值为其长度
    ## 使用单层LSTM 输出的向量维度是50，输入的向量维度是vocab_dim,激活函数relu
    model.add(LSTM(activation="relu", units=50, recurrent_activation="hard_sigmoid"))
    #relu运行速度快，并且可以减缓梯度消失
    print('---'*45)
    model.add(Dropout(0.5))#随机删除网络中的一些隐藏神经元,防止过拟合，实现一定程度的正则化
    model.add(Dense(1))#添加全连接层
    model.add(Activation('sigmoid'))
    print('Compiling the Model...')
    ## 优化函数使用的是adam，收敛效果较好
    model.compile(loss='binary_crossentropy',#对数损失，对于交叉熵，转化为log，计算方便
                  optimizer='adam', metrics=['accuracy'])
    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))

    #plot_model(model, to_file='model.png')
    print("Evaluate...")

    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    yaml_string = model.to_yaml()
    with open('../lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../lstm_data/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
def train():
    print('Loading Data...')
    combined, y = get_datanew()  # 获取数据并分词
    #获取数据

    print('Training a Word2vec model...')
    #获取词向量
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape, x_test.shape)
    print(n_symbols)
    print(embedding_weights)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1,-1)
    model = Word2Vec.load('../lstm_data/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


# 执行结果
def lstm_predict(string):
    print('loading model......')
    with open('../lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)

    # print data
    result = model.predict_classes(data)
    print(result)
    if result[0][0] == 1:
        print(' positive')
    else:
        print(' negative')


if __name__ == '__main__':
    train()
    string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    lstm_predict(string)
