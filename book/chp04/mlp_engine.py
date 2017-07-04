import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Mlp_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir
        self.batch_size = 100
        self.n = 784
        self.k = 10
        self.L = np.array([self.n, 512, self.k])
        self.keep_prob = 0.75
                
    def build_model(self):
        relu_node = 1
        if 1 == relu_node:
            return self.build_relu()
        else:
            return self.build_sigmoid()
        
    def build_relu(self):
        print('###### relu #####')
        X = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])
        #隐藏层
        W_1 = tf.Variable(tf.truncated_normal([784, 512], mean=0.0, stddev=0.1)) #初始化隐含层权重W1，服从默认均值为0，标准差为0.1的截断正态分布
        b_2 = tf.Variable(tf.zeros([512])) #隐含层偏置b1全部初始化为0
        z_2 = tf.matmul(X, W_1) + b_2
        a_2 = tf.nn.relu(z_2)
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        a_2_dropout = tf.nn.dropout(a_2, keep_prob)
        #输出层
        W_2 = tf.Variable(tf.zeros([512, 10]))
        b_3 = tf.Variable(tf.zeros([10]))
        z_3 = tf.matmul(a_2_dropout, W_2) + b_3
        y_ = tf.nn.softmax(z_3)
        #训练部分
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))
        #train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        lanmeda = 0.001
        loss = cross_entropy + lanmeda*(tf.reduce_sum(W_1**2) + tf.reduce_sum(W_2**2))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)
        correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X, y_, y, keep_prob, cross_entropy, train_step, \
                correct_prediction, accuracy

    
    def build_sigmoid(self):
        print('###### sigmoid #####')
        self.keep_prob = 0.90
        X = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, shape=[None, 10])
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        # 隐藏层
        W_1 = tf.Variable(tf.random_normal(shape=[784, 512], mean=0.0, stddev=1.0)) # W_t
        b_2 = tf.Variable(tf.zeros(shape=[512]))
        z_2 = tf.matmul(X, W_1) + b_2
        a_2 = tf.nn.sigmoid(z_2)
        # 输出层
        W_2 = tf.Variable(tf.random_normal(shape=[512, 10], mean=0.0, stddev=1.0)) # W_t
        b_3 = tf.Variable(tf.random_normal(shape=[10], mean=0.0, stddev=1.0))
        z_3 = tf.matmul(a_2, W_2) + b_3
        y_ = tf.nn.softmax(z_3)
        # 代价函数
        # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_, y)
        cross_entropy = tf.reduce_sum(- y * tf.log(y_), 1)
        loss = tf.reduce_mean(cross_entropy)
        #train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)
        # 精度计算
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X, y_, y, keep_prob, cross_entropy, train_step, \
                correct_prediction, accuracy
    
    def train(self, mode=TRAIN_MODE_NEW, ckpt_file='work/lgr.ckpt'):
        X_train, y_train, X_validation, y_validation, X_test, \
                y_test, mnist = self.load_datasets()
        X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        epochs = 10
        saver = tf.train.Saver()
        total_batch = int(mnist.train.num_examples/self.batch_size)
        check_interval = 50
        best_accuracy = -0.01
        improve_threthold = 1.005
        no_improve_steps = 0
        max_no_improve_steps = 3000
        is_early_stop = False
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Mlp_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                for batch_idx in range(total_batch):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    X_mb, y_mb = mnist.train.next_batch(self.batch_size)
                    sess.run(train_step, feed_dict={X: X_mb, y: y_mb, keep_prob: self.keep_prob})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        train_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_train, y: y_train, keep_prob: 1.0})
                        validation_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_validation, y: y_validation, keep_prob: 1.0})
                        if best_accuracy < validation_accuracy:
                            if validation_accuracy / best_accuracy >= \
                                    improve_threthold:
                                no_improve_steps = 0
                            best_accuracy = validation_accuracy
                            saver.save(sess, ckpt_file)
                        print('{0}:{1}# train:{2}, validation:{3}'.format(
                                epoch, batch_idx, train_accuracy, 
                                validation_accuracy))
            print(sess.run(accuracy, feed_dict={X: X_test,
                                      y: y_test, keep_prob: 1.0}))
        
    def run(self, ckpt_file='work/lgr.ckpt'):
        img_file = 'datasets/test6.png'
        img = io.imread(img_file, as_grey=True)
        raw = [1 if x<0.5 else 0 for x in img.reshape(784)]
        sample = np.array(raw)
        X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist = self.load_datasets()
        X, y_, y, keep_prob, cross_entropy, train_step, correct_prediction, \
                accuracy = self.build_model()
        #sample = X_test[102]
        X_run = sample.reshape(1, 784)
        saver = tf.train.Saver()
        digit = -1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_file)
            rst = sess.run(y_, feed_dict={X: X_run, keep_prob: 1.0})
            print('rst:{0}'.format(rst))
            max_prob = -0.1
            for idx in range(10):
                if max_prob < rst[0][idx]:
                    max_prob = rst[0][idx]
                    digit = idx;
        img_in = sample.reshape(28, 28)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
        plt.show()
        
    def load_datasets(self):
        ''' 调用Tensorflow的input_data，读入MNIST手写数字识别数据集的
        训练样本集、验证样本集、测试样本集
        '''
        mnist = input_data.read_data_sets(self.datasets_dir, 
                one_hot=True)
        X_train = mnist.train.images
        y_train = mnist.train.labels
        X_validation = mnist.validation.images
        y_validation = mnist.validation.labels
        X_test = mnist.test.images
        y_test = mnist.test.labels
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist





        
        '''
        print('b1')
        x1 = np.array([[1.0, 2.0],[3.0, 4.0], [5.0, 6.0]])
        print(x1)
        x2 = np.array([8.0, 9.0])
        print(x2)
        x3 = x1 + x2
        print(x3)
        X = tf.placeholder(tf.float32, [None, self.L[0]])
        layers = self.L.shape[0]
        W = []
        b = []
        z = []
        a = []
        for layer in range(layers-1):
            W += [tf.Variable(tf.random_normal([self.L[layer+1], self.L[layer]]))]
        z += [X]
        a += [X]
        b += [tf.Variable(tf.random_normal(self.L[0])]
        for layer in range(1, layers):
            b += [tf.Variable(tf.random_normal(self.L[layer])]
            z += [tf.matmul(a[layer-1], tf.transpose(W[layer-1])) + b[layer-1]]
            a += [tf.nn.relu(z)]
        # 隐藏层定义
        # 回归层定义
        '''
        '''
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        #y_ = tf.matmul(X, W) + b
        z = tf.matmul(X, W) + b
        y_ = tf.nn.softmax(z)
        y = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
        lanmeda = 0.001
        J = cross_entropy + lanmeda*tf.reduce_sum(W**2)
        #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(
        #       cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(J)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X, W, b, y_, y, cross_entropy, train_step, \
                correct_prediction, accuracy
        '''



























