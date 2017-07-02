import tensorflow.python.platform
import numpy as np
import tensorflow as tf
import plot_boundary_on_data
import csv
from seg_ds_loader import Seg_Ds_Loader

# Global variables.
NUM_LABELS = 2    # The number of labels.
BATCH_SIZE = 100  # The number of training examples to use per training step.

# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', 'datasets/linear_data_train.csv',
                           'File containing the training data (labels & features).')
tf.app.flags.DEFINE_string('test', 'datasets/linear_data_eval.csv',
                           'File containing the test data (labels & features).')
tf.app.flags.DEFINE_integer('num_epochs', 5,
                            'Number of examples to separate from the training '
                            'data for the validation set.')
tf.app.flags.DEFINE_boolean('verbose', True, 'Produce verbose output.')
tf.app.flags.DEFINE_boolean('plot', True, 'Plot the final decision boundary on the data.')
FLAGS = tf.app.flags.FLAGS

def main(argv=None):
    # 设置配置参数
    num_epochs = FLAGS.num_epochs # 迭代次数
    verbose = FLAGS.verbose # 打印调试信息
    plot = FLAGS.plot # 绘图
    train_data_filename = FLAGS.train # 训练数据文件
    test_data_filename = FLAGS.test # 测试文件数据
    # 载入训练样本集、验证样本集、测试样本集
    loader = Seg_Ds_Loader()
    train_file, validation_file, test_file = loader.prepare_datesets(train_data_filename, test_data_filename)
    train_data,train_labels = loader.load_dataset(train_file, NUM_LABELS)
    test_data, test_labels = loader.load_dataset(test_file, NUM_LABELS)
    m, n = train_data.shape # m：测试样本集中样本数；n：特征向量维度
    # Get the number of epochs for training.
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    X = tf.placeholder("float", shape=[None, n])
    y_ = tf.placeholder("float", shape=[None, NUM_LABELS])    
    # 定义计算流图
    W = tf.Variable(tf.zeros([n,NUM_LABELS]))
    b = tf.Variable(tf.zeros([NUM_LABELS]))
    z = tf.matmul(X,W) + b
    y = tf.nn.softmax(tf.matmul(X,W) + b)
    # 优化算法
    cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # ??效果好，仅需5次迭代
    #cross_entropy = tf.reduce_sum(tf.pow(y_-y, 2)) / (2*m) # 需要10000次左右迭代才能得到有意义结果
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y)) # 不工作
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # 装入测试样本集
    test_data_node = tf.constant(test_data)
    # 模型评估
    predicted_class = tf.argmax(y,1);
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # 开始训练网络
    with tf.Session() as s:
        # 初始化所有变量
        s.run(tf.global_variables_initializer())        
        # 模型训练
        for epoch in range(num_epochs):
            batch_num = m // BATCH_SIZE
            for batch_idx in range(batch_num):
                offset = batch_idx * BATCH_SIZE
                X_mb = train_data[offset:(offset+BATCH_SIZE), :]
                y_mb = train_labels[offset:(offset+BATCH_SIZE)]
                train_step.run(feed_dict={X: X_mb, y_: y_mb})
        # 训练结果显示
        if verbose:
            print('W:{0}'.format(s.run(W)))
            print('b:{0}'.format(s.run(b)))
            x_t = test_data[:1]
            z_t = s.run(tf.matmul(x_t,W)+b)
            y_t = s.run(tf.nn.softmax(z_t))
            print('Applying model to first test instance: {0}=>{1} {2}'.format(x_t, z_t, y_t))
        # 显示模型评估结果    
        print("Accuracy:", accuracy.eval(feed_dict={X: test_data, y_: test_labels}))
        # 绘制图像
        if plot:
            eval_fun = lambda X: predicted_class.eval(feed_dict={x:X}); 
            plot_boundary_on_data.plot(s.run(W), s.run(b), train_data_filename)
    
if __name__ == '__main__':
    tf.app.run()
