import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class Cnn_Engine(object):
    # 采用习惯用法定义常量
    TRAIN_MODE_NEW = 1
    TRAIN_MODE_CONTINUE = 2
    
    def __init__(self, datasets_dir):
        self.datasets_dir = datasets_dir
        self.batch_size = 100
        self.n = 784
        self.k = 10
        self.L = np.array([self.n, 512, self.k])
        self.lanmeda = 0.001
        self.keep_prob = 0.75
        self.model = {}
                
    def build_model(self):
        print('build convolutional neural network')
        X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        y = tf.placeholder(shape=[None, self.k], dtype=tf.float32)
        keep_prob = tf.placeholder(tf.float32) #Dropout失活率
        # 第2层第1个卷积层c1
        W_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 6], mean=0.0, stddev=0.1))
        b_2 = tf.Variable(tf.zeros([28, 28, 6]))
        z_2 = tf.nn.conv2d(X, W_1, strides=[1, 1, 1, 1], padding='SAME') + b_2
        a_2 = tf.nn.relu(z_2)
        # 第3层第1个最大池化层
        m_3 = tf.nn.max_pool(a_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 第4层第2个卷积层C2
        W_4 = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0.0, stddev=0.1))
        b_4 = tf.Variable(tf.zeros([10, 10, 16]))
        z_4 = tf.nn.conv2d(m_3, W_4, strides=[1, 1, 1, 1], padding='VALID') + b_4
        a_4 = tf.nn.relu(z_4)
        # 第5层第2个最大池化层
        m_5 = tf.nn.max_pool(a_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 第6层第3个卷积层
        W_5 = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 120], mean=0.0, stddev=0.1))
        b_6 = tf.Variable(tf.zeros([1, 1, 120]))
        z_6_raw = tf.nn.conv2d(m_5, W_5, strides=[1, 1, 1, 1], padding='VALID') + b_6
        z_6 = tf.reshape(z_6_raw, [-1, 120])
        a_6 = tf.nn.relu(z_6)
        # 第7层第1个全连接层
        W_6 = tf.Variable(tf.truncated_normal(shape=[120, 84], mean=0.0, stddev=0.1))
        b_7 = b_6 = tf.Variable(tf.zeros([84]))
        z_7 = tf.matmul(a_6, W_6) + b_7
        a_7 = tf.nn.relu(z_7)
        # 第8层第2个全连接层
        W_7 = tf.Variable(tf.truncated_normal(shape=[84, 10], mean=0.0, stddev=0.1))
        b_8 = tf.Variable(tf.zeros([10]))
        z_8 = tf.matmul(a_7, W_7) + b_8
        y_ = tf.nn.softmax(z_8)
        #训练部分
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), 
        reduction_indices=[1]))
        #train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
        loss = cross_entropy + self.lanmeda*(tf.reduce_sum(W_1**2) + 
                tf.reduce_sum(W_4**2) + 
                tf.reduce_sum(W_5**2) + 
                tf.reduce_sum(W_6**2) + 
                tf.reduce_sum(W_7**2))
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(loss)
        correct_prediction = tf.equal(tf.arg_max(y_, 1), tf.arg_max(y, 1))
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
        eval_runs = 0
        eval_times = []
        train_accs = []
        validation_accs = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if Cnn_Engine.TRAIN_MODE_CONTINUE == mode:
                saver.restore(sess, ckpt_file)
            for epoch in range(epochs):
                if is_early_stop:
                    break
                for batch_idx in range(total_batch):
                    if no_improve_steps >= max_no_improve_steps:
                        is_early_stop = True
                        break
                    X_mb_raw, y_mb = mnist.train.next_batch(self.batch_size)
                    X_mb = X_mb_raw.reshape([self.batch_size, 28, 28, 1])
                    sess.run(train_step, feed_dict={X: X_mb, y: y_mb, 
                            keep_prob: self.keep_prob})
                    no_improve_steps += 1
                    if batch_idx % check_interval == 0:
                        eval_runs += 1
                        eval_times.append(eval_runs)
                        train_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_train.reshape([-1, 28, 28, 1]), y: y_train, keep_prob: 1.0})
                        train_accs.append(train_accuracy)
                        validation_accuracy = sess.run(accuracy, 
                                feed_dict={X: X_validation.reshape([-1, 28, 28, 1]), y: y_validation, 
                                keep_prob: 1.0})
                        validation_accs.append(validation_accuracy)
                        if best_accuracy < validation_accuracy:
                            if validation_accuracy / best_accuracy >= \
                                    improve_threthold:
                                no_improve_steps = 0
                            best_accuracy = validation_accuracy
                            saver.save(sess, ckpt_file)
                        print('{0}:{1}# train:{2}, validation:{3}'.format(
                                epoch, batch_idx, train_accuracy, 
                                validation_accuracy))
            print(sess.run(accuracy, feed_dict={X: X_test.reshape([-1, 28, 28, 1]),
                                      y: y_test, keep_prob: 1.0}))
            plt.figure(1)
            plt.subplot(111)
            plt.plot(eval_times, train_accs, 'b-', label='train accuracy')
            plt.plot(eval_times, validation_accs, 'r-', 
                    label='validation accuracy')
            plt.title('accuracy trend')
            plt.legend(loc='lower right')
            plt.show()
        
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
                    digit = idx
            '''
            # W_1_1
            W_1 = sess.run(self.model['W_1'])
            wight_map = W_1[:,0].reshape(28, 28)
            a_2 = sess.run(self.model['a_2'], feed_dict={X: X_run, \
                    keep_prob: 1.0})
            a_2_raw = a_2[0]
            a_2_img = a_2_raw[0:484]
            feature_map = a_2_img.reshape(22, 22)
        img_in = sample.reshape(28, 28)
        plt.figure(1)
        plt.subplot(131)
        plt.imshow(img_in, cmap='gray')
        plt.title('result:{0}'.format(digit))
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(wight_map, cmap='gray')
        plt.axis('off')
        plt.title('wight row')
        plt.subplot(133)
        plt.imshow(feature_map, cmap='gray')
        plt.axis('off')
        plt.title('hidden layer')
        plt.show()
        '''
        
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
        '''
        print('X_train: {0} y_train:{1}'.format(
                X_train.shape, y_train.shape))
        print('X_validation: {0} y_validation:{1}'.format(
                X_validation.shape, y_validation.shape))
        print('X_test: {0} y_test:{1}'.format(
                X_test.shape, y_test.shape))
        image_raw = (X_train[1] * 255).astype(int)
        image = image_raw.reshape(28, 28)
        label = y_train[1]
        idx = 0
        for item in label:
            if 1 == item:
                break
            idx += 1
        plt.title('digit:{0}'.format(idx))
        plt.imshow(image, cmap='gray')
        plt.show()
        '''
        return X_train, y_train, X_validation, y_validation, \
                X_test, y_test, mnist

