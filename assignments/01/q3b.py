import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import tensorflow as tf
import time

import sys
sys.path.append("/Users/williamlane/Desktop")

import utils


def download_data(mnist_folder):
    # utils.download_mnist(mnist_folder)
    train, val, test = utils.read_mnist(mnist_folder, flatten=True)

    return train, val, test


def make_dataset(data, shuffle=None, batch_size=None):
    ds = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        ds = ds.shuffle(shuffle)

    if batch_size:
        ds = ds.batch(batch_size)

    return ds


def make_iterator(dataset):
    iterator = tf.data.Iterator.from_structure(dataset.output_types, 
                                           dataset.output_shapes)

    return iterator


def make_variables(layers):
    Ws = []
    bs = []
    regs = []
    for i in range(1, len(layers)):
        W = tf.get_variable(f"weights_{i}", 
            initializer=tf.random_normal(shape=(layers[i-1], layers[i]), stddev=0.01))
        b = tf.get_variable(f"bias_{i}", initializer=tf.zeros(shape=(1, layers[i])))
        reg = tf.nn.l2_loss(W)

        Ws.append(W)
        bs.append(b)
        regs.append(reg)

    return Ws, bs, regs


def make_network(input_data, activations, layers, Ws, bs):
    out = input_data
    for i, f in enumerate(activations):
        out = f(tf.matmul(out, Ws[i]) + bs[i])

    return out

    # Debug
    # W = tf.get_variable(f"weights_debug", 
    #         initializer=tf.random_normal(shape=(784, 10), stddev=0.01))
    # b = tf.get_variable(f"bias_debug", initializer=tf.zeros(shape=(1, 10)))

    # return tf.nn.sigmoid(tf.matmul(input_data, W) + b)


def make_loss_function(label, classifier, regs=None, lambda_=1e-4):
    loss = tf.losses.softmax_cross_entropy(label, classifier)

    if regs:
        for reg in regs:
            loss += lambda_ * reg

    return loss


def make_adam_optimizer(loss, learning_rate=1e-3):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


def make_test_accuracy_op(classifier, label):
    predictions = tf.argmax(classifier, axis=1)
    correct = tf.equal(predictions, tf.argmax(label, 1))

    return tf.reduce_sum(tf.cast(correct, tf.float32))


def train_model(loss, optimizer, n_epochs, n_test, 
                train_init, test_init, test=False, accuracy=None):
    with tf.Session() as sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())

        # train the model n_epochs times
        for i in range(n_epochs):   
            sess.run(train_init)    # drawing samples from train_data
            
            total_loss = 0
            n_batches = 0
            try:
                while True:
                    _, l = sess.run([optimizer, loss])
                    total_loss += l
                    n_batches += 1
            except tf.errors.OutOfRangeError:
                pass
            print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        
        print('Total time: {0} seconds'.format(time.time() - start_time))

        if test:
            sess.run(test_init)         # drawing samples from test_data
            total_correct_preds = 0
            try:
                while True:
                    accuracy_batch = sess.run(accuracy)
                    total_correct_preds += accuracy_batch
            except tf.errors.OutOfRangeError:
                pass

            print('Accuracy {0}'.format(total_correct_preds/n_test))


def main():
    n_train = 60000
    n_test = 10000
    n_features = 784
    n_output = 10
    n_epochs = 20
    shuffle_size = 1
    batch_size = 128
    
    layers = [n_features, 512, n_output]
    activations = [tf.nn.relu for _ in range(1, len(layers))]
    activations[-1] = tf.nn.softmax
    # print(activations)

    train, val, test = download_data("examples/data/mnist")
    train_data = make_dataset(train, shuffle_size, batch_size)
    test_data = make_dataset(test, shuffle_size, batch_size)

    iterator = make_iterator(train_data)
    train_init = iterator.make_initializer(train_data)
    test_init = iterator.make_initializer(test_data)
    img, label = iterator.get_next()

    Ws, bs, regs = make_variables(layers)

    nn = make_network(img, activations, layers, Ws, bs)
    loss = make_loss_function(label, nn)
    optimizer = make_adam_optimizer(loss, learning_rate=0.002)

    accuracy = make_test_accuracy_op(nn, label)
    train_model(loss, optimizer, n_epochs, n_test,
                train_init, test_init, test=True, accuracy=accuracy) 

if __name__ == "__main__":
    main()