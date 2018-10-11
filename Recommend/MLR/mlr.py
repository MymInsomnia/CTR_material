import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score
import pandas as pd
from fetchdata import getdata


def mlr_test(m, epo):

    """
    MLR模型的丐版实现，未添加L1、L2-1正则并实现在L1、L2-1正则下的参数更新。
    数据集比较曹丹，另L2-1正则和L2正则有多大区别呢？
    @param m: 结构先验个数
    @param epo: 训练轮数
    @return
    """

    m = m
    learning_rate = 0.3

    x = tf.placeholder(tf.float32,  shape=[None,  108])  # 数据入口-x
    y = tf.placeholder(tf.float32,  shape=[None])  # 数据入口-y

    u = tf.Variable(tf.random_normal([108, m], 0.0, 0.5), name='u')  # 初始化向量u
    w = tf.Variable(tf.random_normal([108, m], 0.0, 0.5), name='w')  # 初始化向量w

    U = tf.matmul(x, u)
    p1 = tf.nn.softmax(U)  # 得到结构先验-p1

    W = tf.matmul(x, w)
    p2 = tf.nn.sigmoid(W)  # 得到结构内的预测结果-p2

    pred = tf.reduce_sum(tf.multiply(p1, p2), 1)  # 在每一结构中的预测结果加权相加

    paras = tf.concat([w, u], 0)
    l1_loss = tf.contrib.layers.l1_regularizer(0.1)(paras)  # l1正则项
    l2_loss = tf.contrib.layers.l2_regularizer(0.1)(paras)  # l2正则项

    cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,  labels=y)+l1_loss+l2_loss)

    cost = tf.add_n([cost1])
    train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
    train_x, train_y, test_x, test_y = getdata()
    time_s = time.time()
    result = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0,  epo):
            f_dict = {x: train_x,  y: train_y}

            _,  cost_,  predict_ = sess.run([train_op,  cost,  pred],  feed_dict=f_dict)

            auc = roc_auc_score(train_y,  predict_)
            time_t = time.time()
            if epoch % 100 == 0:
                f_dict = {x: test_x,  y: test_y}
                _,  cost_,  predict_test = sess.run([train_op,  cost,  pred],  feed_dict=f_dict)
                test_auc = roc_auc_score(test_y,  predict_test)
                print("%d %ld cost:%f, train_auc:%f, test_auc:%f" % (epoch,  (time_t - time_s),  cost_,  auc,  test_auc))
                result.append([epoch, (time_t - time_s), auc, test_auc])

    pd.DataFrame(result, columns=['epoch', 'time', 'train_auc', 'test_auc']).to_csv("data/mlr_"+str(m)+'.csv')
