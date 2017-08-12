import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from app_global import appGlobal as ag

class RecommendEngine(object):
    def __init__(self):
        print('initialize recommend engine')
        self.n = 2 # 知识点数量
        self.nm = 5 # 题目总数
        self.nu = 5 # 学生总数
        self.lanmeda = 0.1 # L2调整项系数
        self.epochs = 5000 # 训练遍数
        
    def run(self):
        self.Y_ph, self.r, self.mu = self.load_dataset()
        self.train()
        self.predict(4, 2)
        
    def predict(self, ui, xi):
        print(self.Xv[xi])
        Uv = np.transpose(self.UTv)
        print(Uv[ui])
        print(np.dot(self.Xv[xi], Uv[ui]) + self.mu[xi][0])
        
    def load_dataset(self):
        ph = np.zeros(shape=(self.nm, self.nu), dtype=np.float32)
        r = np.ones(shape=(self.nm, self.nu), dtype=np.int32)
        # first row
        ph[0][0] = 5.0
        r[0][0] = 1
        ph[0][1] = 5.0
        r[0][1] = 1
        ph[0][2] = 0.0
        r[0][2] = 1
        ph[0][3] = 0.0
        r[0][3] = 1
        ph[0][4] = -1.0
        r[0][4] = 0
        # second row
        ph[1][0] = 5.0
        r[1][0] = 1
        ph[1][1] = -1.0
        r[1][1] = 0
        ph[1][2] = -1.0
        r[1][2] = 0
        ph[1][3] = 0.0
        r[1][3] = 1
        ph[1][4] = -1.0
        r[1][4] = 0
        # third row
        ph[2][0] = -1.0
        r[2][0] = 0
        ph[2][1] = 4.0
        r[2][1] = 1
        ph[2][2] = 0.0
        r[2][2] = 1
        ph[2][3] = -1.0
        r[2][3] = 0
        ph[2][4] = -1.0
        r[2][4] = 0
        # forth row
        ph[3][0] = 0.0
        r[3][0] = 1
        ph[3][1] = 0.0
        r[3][1] = 1
        ph[3][2] = 5.0
        r[3][2] = 1
        ph[3][3] = 4.0
        r[3][3] = 1
        ph[3][4] = -1.0
        r[3][4] = 0
        # fifth row
        ph[4][0] = 0.0
        r[4][0] = 1
        ph[4][1] = 0.0
        r[4][1] = 1
        ph[4][2] = 5.0
        r[4][2] = 1
        ph[4][3] = 0.0
        r[4][3] = 1
        ph[4][4] = -1.0
        r[4][4] = 0
        # 求出mu
        mu = np.zeros(shape=(self.nm, 1))
        for row in range(self.nm):
            sum = 0.0
            num = 0
            for col in range(self.nu):
                if 1 == r[row][col]:
                    sum += ph[row][col]
                    num += 1
            mu[row][0] = sum / num
        print(mu)
        print(ph)
        ph = ph - mu
        print(ph)
        return ph, r, mu
        
    def calDeltaY(self, Y, Y_):
        sum = 0.0
        for row in range(self.nm):
            for col in range(self.nu):
                if 1 == self.r[row][col]:
                    sum += (Y[row][col] - Y_[row][col])*(Y[row][col] - Y_[row][col])
        return sum
    
    def build_model(self):
        print('build model')
        self.Y_ = tf.placeholder(shape=[self.nm, self.nu], dtype=tf.float32, name='Y_')
        self.X = tf.Variable(tf.truncated_normal(shape=[self.nm, self.n], mean=0.0, stddev=0.01, seed=1.0), dtype=tf.float32, name='X')
        self.UT = tf.Variable(tf.truncated_normal(shape=[self.n, self.nu], mean=0.0, stddev=0.01, seed=1.0), dtype=tf.float32, name='X')
        self.Y = tf.matmul(self.X, self.UT)
        self.L = self.calDeltaY(self.Y, self.Y_) #tf.reduce_sum((self.Y - self.Y_)*(self.Y - self.Y_))
        self.J = self.L + self.lanmeda*tf.reduce_sum(self.X**2) + self.lanmeda*tf.reduce_sum(self.UT**2)
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, 
                beta2=0.999, epsilon=1e-08, use_locking=False, 
                name='Adam').minimize(self.J)
        
    def train(self):
        self.build_model()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                X, UT, Y, J, train_step = sess.run([self.X, self.UT, self.Y, self.J, self.train_step], feed_dict={self.Y_: self.Y_ph})
                #print(Y)
                print('{0}:{1}'.format(epoch, J))
            self.Xv = X
            self.UTv = UT