'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/


from __future__ import print_function

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


print("n",mnist.train.num_examples); #55000
# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 10
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model

pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax (10,10) + (10,1)

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    pass
    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        arg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        for i  in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            # print('batch_xs,batch_y',batch_xs.shape,batch_ys.shape); # (10,784) (10,10) (batchNo,num)
            _,c,pre = sess.run(fetches=[optimizer,cost,pred],feed_dict={
                x:batch_xs,
                y:batch_ys
            })
            arg_cost += c/total_batch

        if (epoch+1)%display_step==0:
            print('Epoch','%04d'%(epoch+1),'cost',"{:.9f}".format(arg_cost),'pre',pre.shape) #(10,10)

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''


import tensorflow as tf

a = tf.constant(1,shape=[4,2])
b = tf.constant(2,shape=[2])
c = a+b
with tf.Session() as sess:
    print(sess.run([a,b,c]))
