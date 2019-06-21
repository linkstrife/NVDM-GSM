import tensorflow as tf
import numpy as np
import utils
import os

flags = tf.app.flags
flags.DEFINE_string('data_dir', './data/20news', 'The directory of training data.')
flags.DEFINE_float('learning_rate', 5e-5, 'Learning rate for the model.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('n_hidden', 500, 'Number of hidden nodes.')
flags.DEFINE_integer('n_topic', 50, 'Size of the stochastic topic vector.')
flags.DEFINE_integer('n_sample', 1, 'Number of samples.')
flags.DEFINE_integer('vocab_size', 2000, 'Vocabulary size.')
flags.DEFINE_boolean('test', False, 'Process test data.')
flags.DEFINE_string('non_linearity', 'relu', 'Non-linearity of the MLP.')
flags.DEFINE_string('model_type', 'topic', 'Switch between topic and document model.')
FLAGS = flags.FLAGS

class GSM(object):
    def __init__(self, vocab_size, n_hidden, n_topic, n_sample, learning_rate, batch_size, non_linearity, model_type):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self. n_sample = n_sample
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.non_linearity = non_linearity
        self.model_type = model_type

        # 输入文档向量，batch_size x vocab_size
        self.x = tf.placeholder(tf.float32, [None, vocab_size], name='input')
        # mask paddings, 用于序列补0
        self.mask = tf.placeholder(tf.float32, [None], name='mask')

        with tf.variable_scope('Encoder'):
            # 编码文档向量/feed to VAE
            self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity) # encode document embedding
            self.mean = utils.linear(self.enc_vec, self.n_topic, scope='mean') # 均值模块，dim_doc -> dim_topic
            self.log_sigma = utils.linear(self.enc_vec, self.n_topic, matrix_start_zero=True, bias_start_zero=True, scope='logsigma') # 方差模块

            # 多元高斯KL散度, KL(Norm(sigma, miu^2)||Norm(miu0, sigma0^2)) (set as standard Gaussian distribution, miu = 0.0, sigma = 1.0)
            self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.log_sigma - tf.exp(2 * self.log_sigma), 1)
            self.kld = tf.multiply(self.mask, self.kld)

            # 重参数采样模块，epsilon从标准正态分布中采样，每个topic隐藏向量对应一个epsilon向量，共有n_sample x batch_size个
            epsilon = tf.random_normal((self.n_sample * self.batch_size, self.n_topic), 0, 1) # n_sample = 1
            # epsilon_list = tf.split(0, self.n_sample, self.epsilon) # 划分出每个sample的epsilon

            # 解码向量/文档向量, dec_vec/doc_vec
            dec_vec = self.mean + tf.multiply(epsilon, tf.exp(self.log_sigma))

            # we refer to such models that do not directly assign topics to words as document models instead of topic models
            # GSM process, g(x) = softmax(MLP(miu + eps*sigm))
            # batch_size x self.n_topic
            if self.model_type == 'topic':
                self.theta = tf.nn.softmax(utils.linear(dec_vec, self.n_topic, scope='GSM_encode'), name='GSM')
            elif self.model_type == 'document':
                self.theta = dec_vec # remove the softmax of

        with tf.variable_scope('Decoder'):
            #随机初始化
            topic_vec = tf.get_variable('topic_vec', shape=[self.n_topic, self.n_hidden])
            word_vec  = tf.get_variable('word_vec',  shape=[self.vocab_size, self.n_hidden])

            # n_topic x vocab_size (主题-词矩阵)
            self.beta = tf.nn.softmax(tf.matmul(topic_vec, tf.transpose(word_vec)))

            # 注意这里实际上是p(d|theta) = log(theta * beta)，需要输入theta来生成d，表达式就是d = log(theta * beta)
            # 输出就是重构样本d
            if self.model_type == 'topic':
                self.d_given_theta = tf.log(tf.matmul(self.theta, self.beta))
            elif self.model_type == 'document':
                self.d_given_theta = tf.nn.log_softmax(tf.matmul(self.theta, self.beta))

            # 重构误差
            self.reconstruction_loss = -tf.reduce_sum(tf.multiply(self.d_given_theta, self.x), 1)

        # 重构误差 + KL Divergence
        self.objective = self.reconstruction_loss + self.kld

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        full_vars = tf.trainable_variables()
        variable_names = [v.name for v in full_vars]
        print(variable_names)

        enc_var = utils.variable_parser(full_vars, 'Encoder')
        dec_var = utils.variable_parser(full_vars, 'Decoder')

        enc_grad = tf.gradients(self.objective, enc_var)
        dec_grad = tf.gradients(self.objective, dec_var)

        self.optimize_enc = optimizer.apply_gradients(zip(enc_grad, enc_var))
        self.optimize_dec = optimizer.apply_gradients(zip(dec_grad, dec_var))


def train(sess, model,train_url,test_url,batch_size,training_epochs=1000,alternate_epochs=10):
    """train gsm model."""
    # train_set: 维度为1 x vocab_size，每一维是对应的词出现次数, train_count: 训练集的总词数
    train_set, train_count = utils.data_set(train_url)
    test_set, test_count = utils.data_set(test_url)
    # hold-out development dataset, 选取前50篇文档
    dev_set = test_set[:50]
    dev_count = test_count[:50]

    dev_batches = utils.create_batches(len(dev_set), batch_size, shuffle=False)
    test_batches = utils.create_batches(len(test_set), batch_size, shuffle=False)

    for epoch in range(training_epochs):
        # 创建batches，大小为batch_size
        train_batches = utils.create_batches(len(train_set), batch_size, shuffle=True)
        # -------------------------------
        # train
        for switch in range(0, 2):
            if switch == 0:
                optimize = model.optimize_dec
                print_mode = 'updating decoder'
            elif switch == 1:
                optimize = model.optimize_enc
                print_mode = 'updating encoder'
            for i in range(alternate_epochs):
                loss_sum = 0.0
                ppx_sum = 0.0
                kld_sum = 0.0
                word_count = 0
                doc_count = 0
                # 训练每个batch
                for idx_batch in train_batches:
                    '''
                    data_batch: 当前batch的词频向量集合，batch_size*vocab_size
                    count_batch: 当前batch中每篇文档的词数
                    train_set: 训练集
                    train_count: 训练集词数
                    idx_batch: 当前batch
                    mask: 用于某个batch文档不足时做序列对齐
                    '''
                    data_batch, count_batch, mask = utils.fetch_data(
                        train_set, train_count, idx_batch, FLAGS.vocab_size)
                    # input: x = data_batch, mask = mask
                    input_feed = {model.x.name: data_batch, model.mask.name: mask}
                    # return: loss = objective, kld = kld, optimizer = optimize
                    # 以上三者组成feed_dict, 将模型中的tensor映射到具体的值
                    _, (loss, kld) = sess.run((optimize, [model.objective, model.kld]), input_feed)
                    loss_sum += np.sum(loss)
                    kld_sum += np.sum(kld) / np.sum(mask)
                    # 总词数
                    word_count += np.sum(count_batch)
                    # to avoid nan error, 避免0分母
                    count_batch = np.add(count_batch, 1e-12)
                    # per document loss
                    ppx_sum += np.sum(np.divide(loss, count_batch))
                    doc_count += np.sum(mask)
                print_ppx = np.exp(loss_sum / word_count)
                print_ppx_perdoc = np.exp(ppx_sum 
