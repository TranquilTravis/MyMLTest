#import numpy as np
#import pandas as pd
import tensorflow as tf

#Script for Autoencoder

def encoder(trainData,testData):
    dimension = trainData.shape[1]
    hidden1 = int(dimension/10)
    
    # input layer
    input_layer = tf.placeholder("float", [None, dimension], name = 'input_layer')
    
    # first hidden layer
    hidden_1_layer_vals = {
    'weights':tf.Variable(tf.random_normal([dimension,hidden1])),
    'biases':tf.Variable(tf.random_normal([hidden1]))  }
    
    # second hidden layer 
    hidden_2_layer_vals = {
    'weights':tf.Variable(tf.random_normal([hidden1, hidden1])),
    'biases':tf.Variable(tf.random_normal([hidden1]))  }
    
    # output layer 
    output_layer_vals = {
    'weights':tf.Variable(tf.random_normal([hidden1,dimension])),
    'biases':tf.Variable(tf.random_normal([dimension])) }
    
    # multiply output of input_layer wth a weight matrix and add biases
    layer_1 = tf.nn.sigmoid(
           tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),
           hidden_1_layer_vals['biases']))
    
    # multiply output of layer_1 wth a weight matrix and add biases
    layer_2 = tf.nn.sigmoid(
           tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),
           hidden_2_layer_vals['biases']))
    
    # multiply output of layer_2 wth a weight matrix and add biases
    output_layer = tf.matmul(layer_2, output_layer_vals['weights']) + output_layer_vals['biases']
    
    
    # output_true shall have the original data for error calculations
    output_true = tf.placeholder('float', [None, dimension])
    # define our cost function
    meansq = tf.reduce_mean(tf.square(output_layer - output_true))
    # define our optimizer
    learn_rate = 0.1   # how fast the model should learn
    optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)
    
    # initialising stuff and starting the session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # defining batch size, number of epochs and learning rate
    batch_size = 1000  
    hm_epochs =300    # how many times to go through the entire dataset
    tot_num = trainData.shape[0] # total number of data set
    # running the model for a 1000 epochs taking 100 data in batches
    # total improvement is printed out after each epoch
    for epoch in range(hm_epochs):
        epoch_loss = 0    # initializing error as 0
        for i in range(int(tot_num/batch_size)):
            epoch_x = trainData[ i*batch_size : (i+1)*batch_size ]
            _, c = sess.run([optimizer, meansq],\
                   feed_dict={input_layer: epoch_x, \
                   output_true: epoch_x})
            epoch_loss += c
        print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)
    
    output_train = sess.run(layer_1,\
                   feed_dict={input_layer: trainData})
    
    output_test = sess.run(layer_1,\
                   feed_dict={input_layer: testData})
    
    return output_train,output_test

