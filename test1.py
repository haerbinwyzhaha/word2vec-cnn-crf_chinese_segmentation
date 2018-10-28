import tensorflow as tf
## tensorboard usage test
import  re
import  numpy as np
import word2vec
import json
import cnn_seg
if __name__ == '__main__':
    t = tf.constant([[[1, 2, 3]]])
    t = tf.pad(t,[ [0,1] ,[0,1],[0,1] ],'CONSTANT')
    sess=tf.Session()
    print(sess.run(t))



    # path = './corpus_data/msr_test_gold.utf8'
    # pure_txts,pure_tags,pure_words=word2vec.get_corpus_words(path)
    #
    # print(pure_words[:2],pure_txts[:2])
    #
    # dictionary=json.load(open('dictionary.json','r'))
    # tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    # data=cnn_seg.generate_batch(pure_txts,pure_tags,dictionary,tag2vec)
    # for x,y in data:
    #     print(x)
    # path='./corpus_data/msr_test_gold.utf8'
    # pure_txts,pure_tags=word2vec.get_corpus(path)
    # dictionary=json.load(open('dictionary.json','r'))
    # tag2vec = {'S': [1, 0, 0, 0], 'B': [0, 1, 0, 0], 'M': [0, 0, 1, 0], 'E': [0, 0, 0, 1]}
    # data=cnn_seg.generate_data(pure_txts,pure_tags,dictionary,tag2vec,batch_size=512)
    # i=0
    # for x,y in data:
    #     i+=1
    #     print(len(x))





    # type=1
    #
    # matrix1 = tf.constant([[1,2]], name='matrix1',dtype=tf.float32) #(2,1)
    # matrix2 = tf.constant([[4,3]], name='matrix2') #(1,2)
    # # vari1=tf.Variable(tf.random_uniform([1],-1.0,1.0))
    # w1=tf.constant([[[1],[1]],[[1],[1]]],dtype=tf.float32)
    # print(matrix1.shape)
    # arg_test=tf.cast(tf.equal(tf.argmax(matrix2,1),tf.argmax(matrix1,1)),dtype=tf.float32)
    # # print(w1.shape)
    # # con1d=tf.nn.conv1d(matrix1,w1,padding='SAME',stride=1)
    #
    # input_img = tf.constant([[[[1], [2], [3]],
    #                           [[4], [5], [6]],
    #                           [[7], [8], [9]]]], tf.float32, [1, 3, 3])
    # conv_filter1 = tf.constant([[[[2]]]], tf.float32, [3, 3, 3])
    # op1 = tf.nn.conv1d(input_img, conv_filter1, stride=1, padding='SAME')
    #
    # with tf.Session() as sess:
    #     # results=sess.run(op1)
    #     results=sess.run(arg_test)
    #     print(results)
    # path='./corpus_data/msr_test_gold.utf8'
    # # pure_txts,pure_tag=word2vec.get_corpus(path)
    # stops = u'，。！？；、：,\.!\?;:\n'
    # pure_word=[]
    # with open(path,'r') as f:
    #     txt=[line.strip() for line in re.split('['+stops+']',f.read()) if line.strip(' ')]
    #     for line in txt:
    #         pure_word.append([word for word in re.split(' +',line)])
    #
    # indexs=[len(i) for i in pure_word]
    # indexs=np.argsort(indexs)[::-1]
    # pure_word=[pure_word[i] for i in indexs]
    # print(pure_word[:2])
    #
    # pure_txts=[]
    # pure_tags=[]
    # for i in range(len(pure_word)):
    #     pure_txts.append('')
    #     pure_tags.append('')
    #     for word in pure_word[i]:
    #         pure_txts[-1]+=word
    #         pure_tags[-1]+=word2vec.makelabel(word)
    # print(pure_txts[:2])
    # print(pure_tags[:2])

    # print(pure_txts)