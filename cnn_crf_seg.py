import  json
import numpy as np
import tensorflow as tf
import word2vec
import tqdm

def generate_data(pure_txts,pure_tags,dictionary,tag2vec,batch_size=256):
    # batch_size = 256
    l=len(pure_txts[0])
    x=[]
    y=[]
    lengths=[]
    for i in range(len(pure_txts)):
        if len(pure_txts[i])!=l or len(x)==batch_size:
            yield x,y,lengths
            x=[]
            y=[]
            lengths=[]
            l=len(pure_txts[i])
        x.append([dictionary.get(j,0) for j in pure_txts[i]])
        y.append([tag2vec[j] for j in pure_tags[i]])
        lengths+=[len(pure_txts[i])]

def generate_batch(pure_txts,pure_tags,dictionary,tag2vec):
    for i in range(len(pure_txts)):
        x=[dictionary.get(word,0) for word in pure_txts[i]]
        y=[tag2vec.get(tag) for tag in pure_tags[i]]
        lengths=[len(pure_txts[i])]
        yield [x], [y],lengths

def seg_word(txt,pred):
    begin,end=0,0
    words=[]
    for index,char in enumerate(txt):
        if pred[index]==1:
            begin=index
        elif pred[index]==3:
            words.append(txt[begin:index+1])
            end=index+1
        elif pred[index]==0:
            words.append(char)
            end=index+1
    if end < len(txt)-1 :
        words.append(txt[end:])
    return words

def accuracy_p(pre_segs,pure_words):
    accuracy=float(0)
    for i in range(len(pure_words)):
        temp_words=pure_words[i]
        temp_seg=pre_segs[i]
        right=0
        for word in temp_seg:
            if word in temp_words:
                right+=1
        accuracy+=right/len(temp_seg)
    accuracy/=len(pre_segs)
    return accuracy
def accuracy_r(pre_segs,pure_words):
    accuracy=float(0)
    for i in range(len(pure_words)):
        temp_words=pure_words[i]
        temp_seg=pre_segs[i]
        right=0
        for word in temp_seg:
            if word in temp_words:
                right+=1
        accuracy+=right/len(temp_words)
    accuracy/=len(pre_segs)
    return accuracy
def compute_f(pre_segs,pure_words):
    p=accuracy_p(pre_segs,pure_words)
    r=accuracy_r(pre_segs,pure_words)
    return (p*r*2)/(p+r)

def cnn_crf_model(pure_txts,pure_tags,dictionary,tag2vec,hidden_size=512,train=True,nums_step=300,keep_prob=0.5):
    embedding_size=128
    # batchsize,seqlength
    x=tf.placeholder(dtype=tf.int32,shape=[None,None])
    # batchsize,seqlength
    y=tf.placeholder(dtype=tf.int32,shape=[None,None])

    batch_size=tf.shape(x)[0]
    nums_word=tf.shape(x)[1]

    sequence_length=tf.placeholder(dtype=tf.int32,shape=[None])

    embedding = word2vec.word2vec_train(vocabulary_size=len(dictionary), type=0)
    embedded=tf.nn.embedding_lookup(embedding,x)

    ##define CNN Convolution
    W1=tf.Variable(tf.random_uniform([3,embedding_size,hidden_size],-1,1))
    W1=tf.nn.dropout(W1,keep_prob=keep_prob)
    b1=tf.Variable(tf.zeros([hidden_size]))
    a1=tf.nn.relu(tf.nn.conv1d(embedded,W1,stride=1,padding='SAME')+b1)

    W2=tf.Variable(tf.random_uniform([3,hidden_size,int(hidden_size/2)],-1,1))
    W2 = tf.nn.dropout(W2, keep_prob=keep_prob)
    b2=tf.Variable(tf.zeros([int(hidden_size/2)]))
    a2=tf.nn.relu(tf.nn.conv1d(a1,W2,stride=1,padding='SAME')+b2)

    W3=tf.Variable(tf.random_uniform([3,int(hidden_size/2),embedding_size],-1,1))
    W3 = tf.nn.dropout(W3, keep_prob=keep_prob)
    b3=tf.Variable(tf.zeros([embedding_size]))
    a3=tf.nn.relu(tf.nn.conv1d(a2,W3,stride=1,padding='SAME')+b3)

    ##  基于crf、计算相应loss
    w=tf.Variable(tf.random_uniform([embedding_size,4],-1,1))
    b=tf.Variable(tf.zeros([4]))

    calculate_x=tf.reshape(a3,shape=[-1,embedding_size])
    scores=tf.matmul(calculate_x,w)+b
    reshape_scores=tf.reshape(scores,shape=[batch_size,nums_word,4])

    log_likelihood,tran_parms=tf.contrib.crf.crf_log_likelihood(reshape_scores,y,sequence_length)

    loss=tf.reduce_mean(-log_likelihood)

    train_step=tf.train.AdamOptimizer().minimize(loss)
    ##decode get prediction
    decode_tag,_=tf.contrib.crf.crf_decode(reshape_scores,tran_parms,sequence_length)

    correct_prediction=tf.equal(decode_tag,y)

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))

    init=tf.global_variables_initializer()

    sess=tf.Session()

    if train:
        sess.run(init)
        for epoch in range(nums_step):
            train_data=tqdm.tqdm(generate_data(pure_txts,pure_tags,dictionary,tag2vec))
            k=0
            accs=[]
            for train_x,train_y,train_length in train_data:
                if k%100==0:
                    acc=sess.run(accuracy,feed_dict={x:train_x,y:train_y,sequence_length:train_length})
                    train_data.set_description(u'Epoch %s Accuracy: %s'%(epoch+1,acc))
                    accs.append(acc)
                sess.run(train_step,feed_dict={x:train_x,y:train_y,sequence_length:train_length})
            print(u'Epoch %s Mean Accuracy: %s'%(epoch+1,np.mean(accs)))
        saver=tf.train.Saver()
        saver.save(sess,'./model/cnn_crf_seg')
    else :
        saver=tf.train.Saver()
        saver.restore(sess,'./model/cnn_crf_seg')
        data=generate_batch(pure_txts,pure_tags,dictionary,tag2vec)
        pred_tags=[]
        for xx,yy,zz in data:
            predicition_scores=sess.run(reshape_scores,feed_dict={x:xx,y:yy,sequence_length:zz})
            transition_params=sess.run(tran_parms,feed_dict={x:xx,y:yy,sequence_length:zz})
            predicition,_ = tf.contrib.crf.viterbi_decode(predicition_scores[0],transition_params)
            pred_tags.append(predicition)
        return pred_tags

if __name__ == '__main__':
    ## get dictionary
    dictionary=json.load(open('dictionary.json','r'))
    ## get txts and tags
    tag2vec={'S':0,'B':1,'M':2,'E':3}
    vec2tag={0:'S',1:'B','M':2,'E':3}
    train_path='./corpus_data/msr_training.utf8'
    test_path='./corpus_data/msr_test_gold.utf8'

    train_pure_txts,train_pure_tags=word2vec.get_corpus(train_path)
    test_pure_txts,test_pure_tags,test_pure_words=word2vec.get_corpus_words(test_path)

    # 模型训练
    # cnn_crf_model(train_pure_txts,train_pure_tags,dictionary,tag2vec)

    ## 模型预测
    predictions=cnn_crf_model(test_pure_txts,test_pure_tags,dictionary,tag2vec,train=False,keep_prob=1)

    ## 分词
    seg_sequences=[]
    for i in range(len(test_pure_txts)):
        seg_sequences.append(seg_word(test_pure_txts[i],predictions[i]))

    # 模型在测试集上P值 : 0.9287944531959634
    print("模型在测试集上P值 : %s"%(compute_f(seg_sequences,test_pure_words)))

    print_line=5
    for i in range(print_line):
        # print('分词后 :%s'%(seg_word(test_pure_txts[i],predictions[i])))
        print('原句 : %s'%(test_pure_txts[i]))
        print('分词后 :%s' % (seg_word(test_pure_txts[i], predictions[i])),'\n')
