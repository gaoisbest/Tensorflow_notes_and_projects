# Basic concepts
- For untrainable variable (e.g., `global_step` for keeping track of the number of training loop), set `trainable=False`. That is
`global_step = tf.Variable(0, trainable=False, dtype=tf.int32) increment_step = global_step.assign_add(1)`
- We can modify the gradients by using `optimizer.compute_gradients()` and `optimizer.apply_gradients()`. And using `tf.gradient()` to only train part of the network.
- [Comparisions of optimizers](http://sebastianruder.com/optimizing-gradient-descent/).
  - RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. 
  - RMSprop is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. 
  - Adam adds bias-correction and momentum to RMSprop.
- Higher batch size typically requires more epochs since it does fewer update steps. See [Bengio's practical tips](https://arxiv.org/pdf/1206.5533v2.pdf).

# Codes
- Linear regression and polynomial regression
```
%matplotlib inline
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

####################### dummy training data ##############################
n_samples = 100
x_dummy = np.linspace(-3, 3, n_samples)

# linear function
y_dummy = 3 * x_dummy + 5 + np.random.uniform(-5, 5, n_samples) # with random noise

# sin() function
# y_dummy = np.sin(x_dummy) + np.random.uniform(-3, 3, n_samples)
####################### dummy training data ##############################


####################### two version placeholders #########################
# ORIGINAL version: without shape
X = tf.placeholder(dtype=tf.float32, name='input_X')
Y = tf.placeholder(dtype=tf.float32, name='input_y')

# SHAPE version: each input has the explicit shape
# X = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input_X')
# Y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='input_y')
####################### two version placeholders #########################


############################ score function ##############################
# linear regression
W = tf.Variable(initial_value=tf.random_normal([1]), name='weight')
b = tf.Variable(initial_value=tf.random_normal([1]), name='bias')
Y_pred = tf.add(tf.multiply(X, W), b)

# polynomial regression
# Y_pred = tf.Variable(tf.random_normal([1]), name='y_pred')
# for pow_i in range(1, 6):
#    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
#    Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)
############################ score function ##############################


############################ loss function ###############################
# mean squared error loss
# reduce_mean == reduce_sum / number_of_element
# tf.pow(a - b, 2) == tf.squared_difference(a, b). 
#loss = tf.reduce_mean(tf.squared_difference(Y_pred, Y))
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / n_samples


#parameters = tf.trainable_variables()
#loss = tf.add(loss, tf.multiply(1e-6, tf.reduce_sum(tf.square(parameters)))) # L2 norm, ridge regression
#loss = tf.add(loss, tf.multiply(1e-6, tf.reduce_sum(tf.abs(parameters)))) # L1 norm, lasso regression

'''
# Huber loss
# see
def huber_loss(true_labels, predictions, delta=1.0):
    residual = tf.abs(true_labels - predictions)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.4 * tf.square(delta)
    return tf.where(condition, small_res, large_res)
loss = tf.reduce_sum(huber_loss(Y, Y_pred)) / 2*n_samples
'''
############################ loss function ###############################


############################ optimization ###############################
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss)

############################ optimization ###############################


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10):
        for (x, y) in zip(x_dummy, y_dummy):
            sess.run(optimizer, feed_dict={X:x, Y:y})            
            # SHAPE version
            # sess.run(optimizer, feed_dict={X:x.reshape((1, 1)), Y:y.reshape((1, 1))})
        
        tmp_loss = sess.run(loss, feed_dict={X:x_dummy, Y:y_dummy})
        # SHAPE version
        # tmp_loss = sess.run(loss, feed_dict={X:x_dummy.reshape((100,1)), Y:y_dummy.reshape((100,1))})
        
        print('loss:{0}'.format(tmp_loss))
    plt.scatter(x_dummy, y_dummy, marker='o', c='r', linewidths=1)
    
    plt.plot(x_dummy, Y_pred.eval(feed_dict={X:x_dummy}), 'g')
    
    # SHAPE version
    # plt.plot(x_dummy, Y_pred.eval(feed_dict={X:x_dummy.reshape((100,1))}), 'g') 
```
