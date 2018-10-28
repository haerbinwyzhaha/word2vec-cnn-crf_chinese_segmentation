import  tensorflow as tf
import  numpy as np
import word2vec
import json
import tqdm
def generate_batch(pure_txts,pure_tags,dictionary,tag2vec):
    for i in range(len(pure_txts)):
        x=[dictionary.get(word,0) for word in pure_txts[i]]
        y=[tag2vec.get(tag) for tag in pure_tags[i]]
        yield [x], [y]

def generate_data(pure_txts,pure_tags,dictionary,tag2vec,batch_size=256):
    # batch_size = 256
    l=len(pure_txts[0])
    x=[]
    y=[]
    for i in range(len(pure_txts)):
        if len(pure_txts[i])!=l or len(x)==batch_size:
            yield x,y
            x=[]
            y=[]
            l=len(pure_txts[i])
        x.append([dictionary.get(j,0) for j in pure_txts[i]])
        y.append([tag2vec[j] for j in pure_tags[i]])


def viterbi(prediction,trans_pros):
    nodes=[dict(zip( ('S','B','M','E'),pro ) )for pro in prediction]
    trans_pros={state:np.log(num) for state,num in trans_pros.items()}
    paths=nodes[0]
    for t in range(1,len(nodes)):
        path_old=paths.copy()
        paths={}
        for i in nodes[t]:
            new_path={}
            for j in path_old:
                if j[-1]+i in trans_pros:
                    new_path[j+i]=path_old[j]+nodes[t][i]+trans_pros[j[-1]+i]
            pro,key=max([(new_path[key],key)for key,value in new_path.items()])
            paths[key]=pro
    best_pro,best_path=max([ (paths[key],key) for key,value in paths.items() ])
    return  best_path

def generate_all(pure_txts,pure_tags,dictionary,tag2vec):
    x=[]
    y=[]
    for i in range(len(pure_txts)):
        x.extend([dictionary.get(word,0) for word in pure_txts[i]])
        y.extend([tag2vec.get(tag) for tag in pure_tags[i]])

    return [x],[y]
def cnn_model(pure_txts,pure_tags,dictionary,tag2vec,nums_step=10,flag=1):
    #CNN模型实现. 定义embedding, 卷积层,LOSS,optimizer等
    embedding_size=128
    x=tf.placeholder(tf.int32,shape=[None,None])
    y=tf.placeholder(tf.float32,shape=[None,None,4])

    ## 读取训练后的word2vec的embedding
    embedding = word2vec.word2vec_train(vocabulary_size=len(dictionary), type=0) #(len,128)
    ## 获取字嵌入向量
    embedded=tf.nn.embedding_lookup(embedding,x)  #(len(x),128)

    test_input=[[dictionary.get(word,0) for word in pure_txts[0]]]
    test_label=[[tag2vec.get(tag) for tag in pure_tags[0]]]

    ## define w1,b1...
    W1=tf.Variable(tf.random_uniform([3,embedding_size,embedding_size],-1.0,1.0))
    b1=tf.Variable(tf.random_uniform([embedding_size],-1.0,1.0))
    a1=tf.nn.relu(tf.nn.conv1d(embedded,W1,stride=1,padding='SAME')+b1)
    W2=tf.Variable(tf.random_uniform([3,embedding_size,int(embedding_size/4)],-1.0,1.0))
    b2=tf.Variable(tf.random_uniform([int(embedding_size/4)],-1.0,1.0))
    a2=tf.nn.relu(tf.nn.conv1d(a1,W2,padding='SAME',stride=1)+b2)
    W3=tf.Variable(tf.random_uniform([3,int(embedding_size/4),4],-1.0,1.0))
    b3=tf.Variable(tf.random_uniform([4],-1.0,1.0))
    ## softmax compute y_prediction
    y_pre=tf.nn.softmax(tf.nn.conv1d(a2,W3,padding='SAME',stride=1)+b3)

    ## loss- cross entropy, optimizer- adam
    loss=-tf.reduce_mean(y*tf.log(y_pre+1e-20))
    optimizer=tf.train.AdamOptimizer().minimize(loss)

    ## define accuracy
    correct_pre=tf.equal(tf.argmax(y_pre,2),tf.argmax(y,2))
    accuracy=tf.reduce_mean(tf.cast(correct_pre,dtype=tf.float32))

    init=tf.global_variables_initializer()

    if flag==1:
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(nums_step):
                train_data = tqdm.tqdm(generate_data(pure_txts, pure_tags,dictionary,tag2vec))
                k = 0
                accs = []
                for x_inputs, y_inputs in train_data:
                    k += 1
                    if k % 100 == 0:
                        score = sess.run(accuracy, feed_dict={x: x_inputs, y: y_inputs})
                        accs.append(score)
                        train_data.set_description('Epcho %s, Accuracy: %s' % (epoch + 1, score))
                    sess.run(optimizer, feed_dict={x: x_inputs, y: y_inputs})
                print(u'Epoch %s Mean Accuracy : %s' % (epoch + 1, np.mean(accs)))
            saver = tf.train.Saver()
            saver.save(sess=sess, save_path='./model/cnn_seg')
    elif flag==2 :
        ##extract model parameters
        with tf.Session() as sess:
            saver=tf.train.Saver()
            saver.restore(sess=sess,save_path='./model/cnn_seg')
            test_data=generate_batch(pure_txts,pure_tags,dictionary,tag2vec)
            y_pres=[]
            for x_inputs,y_inputs in test_data:
                prediction=sess.run(y_pre,feed_dict={x:x_inputs})
                y_pres.append(prediction[0,:,:])
            return  y_pres
    else :
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path='./model/cnn_seg')
            test_inputs,test_labels=generate_all(pure_txts,pure_tags,dictionary,tag2vec)
            acc=sess.run(accuracy,feed_dict={x:test_inputs,y:test_labels})
            print("(未解码) Accuracy : %s"%(acc))

def segword(txt,best_path):
    begin,end=0,0
    seg_word=[]
    # print(txt)
    # print(best_path)
    for index,char in enumerate(txt):
        signal=best_path[index]
        if signal=='B':
            begin=index
        elif signal=='E':
            seg_word.append(txt[begin:index+1])
            end=index+1
        elif signal=='S':
            seg_word.append(char)
            end=index+1
    if end<len(txt):
        seg_word.append(txt[end:])
    return seg_word

def seg_txt(pure_txts,prediction,trans_pros):
    seg_txt=[]
    for i in range(len(pure_txts)):
        txt=pure_txts[i]
        pre=prediction[i]
        print(txt,pre)
        best_path=viterbi(pre,trans_pros)
        line=segword(txt,best_path)
        seg_txt.append(line)
    return seg_txt

def accuracy_judge(pre_segs,pure_words):
    accuracy=float(0)
    for i in range(len(pure_words)):
        temp_words=pure_words[i]
        temp_seg=pre_segs[i]
        right=0
        for word in temp_seg:
            if word in temp_words:
                right+=1
        accuracy+=right/len(pure_words[i])
    accuracy/=len(pure_words)
    return accuracy

if __name__ == '__main__':
    ##10-15:读取文件
    # path='./corpus_data/msr_test_gold.utf8' # test-set
    # train_path='./corpus_data/msr_training.utf8'
    # train_pure_txts,train_pure_tags=word2vec.get_corpus(train_path)
    # 读取字典文件
    dictionary=json.load(open('dictionary.json','r'))

    tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}

    trans_pro={'SS':1,'BM':1,'BE':1,'SB':1,'MM':1,'ME':1,'EB':1,'ES':1}

    # ##模型训练
    # cnn_model(train_pure_txts,train_pure_tags,dictionary,tag2vec,nums_step=1001)
    #
    ##模型预测
    test_path='./corpus_data/msr_test_gold.utf8'

    test_pure_txts,test_pure_tags,test_pure_words=word2vec.get_corpus_words(test_path)

    prediction = cnn_model(test_pure_txts, test_pure_tags, dictionary, tag2vec, flag=2)
    ## 分词后的文本
    segtxt = seg_txt(test_pure_txts, prediction, trans_pro)

    print("解码后准确率 : %s"%(accuracy_judge(segtxt, test_pure_words)))


