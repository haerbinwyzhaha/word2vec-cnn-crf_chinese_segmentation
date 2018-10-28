###模型思路:利用word2vec 字向量嵌入+纯CNN模型 实现中文分词
import tensorflow as tf
import re
import numpy as np
from collections import  Counter,defaultdict,deque
import random
import tensorflow as tf
import math
import tqdm
import json
def makelabel(words):
    if len(words)==1:
        return 'S'
    else :
        return 'B'+(len(words)-2)*'M'+'E'

def get_corpus(path):
    stops=u'，。！？；、：,\.!\?;:\n'
    pure_txts=[]
    pure_tags=[]
    with open(path,'r') as f:
        txt=[line.strip(' ') for line in re.split('['+stops+']',f.read()) if line.strip(' ')]
        i=0
        for line in txt:
            pure_txts.append('')
            pure_tags.append('')
            for word in re.split(' +',line):
                pure_txts[-1]+=word
                pure_tags[-1]+=makelabel(word)
    #把句子从大到小排序
    indexs=[len(i) for i in pure_txts]
    indexs=np.argsort(indexs)[::-1]
    pure_txts=[pure_txts[i] for i in indexs]
    pure_tags=[pure_tags[i] for i in indexs]
    return pure_txts,pure_tags

def get_corpus_words(path):
    stops = u'，。！？；、：,\.!\?;:\n'
    pure_word = []
    with open(path, 'r') as f:
        txt = [line.strip() for line in re.split('[' + stops + ']', f.read()) if line.strip(' ')]
        for line in txt:
            pure_word.append([word for word in re.split(' +', line)])

    indexs = [len(i) for i in pure_word]
    indexs = np.argsort(indexs)[::-1]
    pure_word = [pure_word[i] for i in indexs]
    pure_txts = []
    pure_tags = []
    for i in range(len(pure_word)):
        pure_txts.append('')
        pure_tags.append('')
        for word in pure_word[i]:
            pure_txts[-1] += word
            pure_tags[-1] += makelabel(word)

    return  pure_txts,pure_tags,pure_word

def make_dic(pure_txts):
    min_count=2
    word_count=Counter(''.join(pure_txts))
    word_count=Counter({word:index for word,index in word_count.items() if index>=min_count})
    id=0
    word_id=defaultdict(int)
    for i in word_count.most_common():
        id+=1
        word_id[i[0]]=id
    # print(word_count)
    return word_id

def make_dict(pure_txts,vacabulary_size,flag=False):
    words=''.join(pure_txts)
    count = [['UNK', -1]]
    count.extend(Counter(''.join(words)).most_common(vacabulary_size-1))
    print(len(count))
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data=list()
    unk_count=0
    # 生成字在词典中对应的位置
    for word in words:
        index=dictionary.get(word,0)
        if index==0:
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverser_dictionary=dict(zip(dictionary.values(),dictionary.keys()))
    if flag:
        json.dump(dictionary, open('dictionary.json', 'w'))
    return data,count,dictionary,reverser_dictionary,len(count)


def generate_batch(data,batch_size=8,nums_skip=2,skip_window=1):
    global data_index
    batch=np.ndarray(shape=(batch_size),dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1
    windows_words=deque(maxlen=3)
    if data_index+span>len(data):
        data_index=0
    windows_words.extend(data[data_index:data_index+span])
    data_index+=span
    for i in range(batch_size//nums_skip):
        words_index=[w for w in range(span) if w!= skip_window]
        for j,word in enumerate(words_index):
            batch[i*nums_skip+j]=windows_words[skip_window]
            labels[i*nums_skip+j,0]=windows_words[word]
        #假如滑到最后
        if data_index==len(data):
            windows_words.extend(data[0:span])
            data_index=span
        else :
            windows_words.append(data[data_index])
            data_index+=1
    # print("索引 ："+str(data_index))
    data_index=(data_index+len(data)-span)%len(data)
    # print("索引 ：" + str(data_index))
    return batch,labels

def word2vec_train(data=None,vocabulary_size=None,learning_rate=1.0,nums_steps=100,type=1):
    batch_size, nums_skip, skip_window = 8, 2, 1
    embedding_size = 128
    num_sampled = 8  # 负采样个数.

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedded = tf.nn.embedding_lookup(embedding, train_inputs)

    nce_w = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_b = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(
        weights=nce_w, biases=nce_b, labels=train_labels,
        inputs=embedded, num_classes=vocabulary_size, num_sampled=num_sampled
    ))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))

    normalized_embeddings = embedding / norm

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # nums_steps = 1000

    with tf.Session() as sess:
        if type==1:
            sess.run(init)

            average_loss = []

            for time in range(nums_steps):
                generate_data = tqdm.tqdm(generate_batch(data), desc=u'Epoch %s ' % (time + 1))
                batch_inputs, batch_labels = generate_data
                _, time_loss = sess.run([optimizer, loss],
                                        feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
                average_loss.append(time_loss)
                if (time + 1) % 1000 == 0:
                    print(u'Epcho %s Mean Loss: %s' % (time + 1, np.mean(average_loss)))
            # embedded_vector=normalized_embeddings.eval()
            normalized_embeddings.eval()

            print(u'Final Mean Loss: %s' % (np.mean(average_loss)))

            saver.save(sess=sess, save_path='./model/word2vec_model')
        else :
            saver.restore(sess,save_path='./model/word2vec_model')
            return normalized_embeddings.eval()





if __name__ == '__main__':
    pass
    ## 获取字典 并存入json文件
    # path1='./corpus_data/msr_training.utf8'
    # path2='./corpus_data/msr_test_gold.utf8'
    # more_txts,more_tags=get_corpus(path2)
    # pure_txts,pure_tags=get_corpus(path1)
    #
    # pure_txts.extend(more_txts)
    # pure_tags.extend(more_tags)
    #
    # max_vacabulary_size=10000
    # data,count,dictionary,reverser_dictionary,vocabulary_size=make_dict(pure_txts,max_vacabulary_size,flag=True)
    # print('Most Common word:',count[:5])
    # print('part of data',data[:8],[reverser_dictionary[i] for i in data[:10]])
    # print(dictionary.items())
    # data_index=0

    # word2vec_train(data,vocabulary_size,nums_steps=10000)






