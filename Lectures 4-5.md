Distributed Representations of Words and Phrases and their Compositionality
Efficient Estimation of Word Representations in Vector Space


Statistically it has the effect that CBOW smoothes over a lot of the distributional
information (by treating an entire context as one observation), this
turns out to be a useful thing for **smaller datasets**. However, skip-gram treats each
context-target pair as a new observation, and this tends to do better when we have **larger
datasets**.


`tf.name_scope` tell TensorBoard to know which nodes should be grouped together.


TensorBoard has two kinds of edges: the solid lines and the dotted
lines. The solid lines represent data flow edges. 
The dotted arrows represent control dependence edges.

Model should be build as a **class**.


The topics we cover today include: tf.train.Saver() class,
TensorFlow’s random seed and NumPy’s random state, and visualization our training
progress (aka more TensorBoard).

`tf.train.Saver()`: periodically save the model’s parameters (i.e., graph’s variables) after a certain number of steps.

```
saver = tf.train.Saver()
with tf.Session() as sess:
    for step in range(training_steps):
	    sess.run([optimizer])
		if (step + 1) % 1000 == 0:
			# model_name- + global_step
		    saver.save(sess, 'checkpoint_directory/model_name', global_step=model.global_step) 
```

**Checkpoint**: the step at which you save your graph’s variables.
```
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
# pass global_step as a parameter to the optimizer
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
```







Variable creation: must pass a Tensor as its **initial value** to the tf.Variable() constructor.

Calling `tf.Variable()` adds three ops to the graph: [1] A **variable op** that holds the variable value. [2] An **initializer op** (actually `tf.assign` op)that sets the variable to its initial value. [3] The ops for the initial value, such as the **zeros op**.

A variable can be pinned to a particular device when it is created, using a `with tf.device(...)`.
```
# Pin a variable to CPU.
with tf.device("/cpu:0"):
  v = tf.Variable(...)

# Pin a variable to GPU.
with tf.device("/gpu:0"):
  v = tf.Variable(...)
```


Saving Variables

Create a Saver with tf.train.Saver() to manage all variables in the model.

# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  ..
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in file: %s" % save_path)
  
# Create some variables.
v1 = tf.Variable(..., name="v1")
v2 = tf.Variable(..., name="v2")
...
# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Do some work with the model
  ...

  reference:

https://www.tensorflow.org/programmers_guide/variables

  
```
tf.nn.embedding_lookup(
    params,
    ids,
    partition_strategy='mod',
    name=None,
    validate_indices=True,
    max_norm=None
)
```
# Looks up `ids` in a list of embedding tensors `params`.

reference:

https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do/41922877




`tf.summary.histogram(name, values, collections=None)`
#Outputs a Summary protocol buffer with a histogram.

`tf.summary.scalar(name, tensor, collections=None)`
#Outputs a Summary protocol buffer containing a single scalar value.

`tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)`
# Merges all summaries collected in the default graph.
reference:
https://www.tensorflow.org/api_docs/python/tf/summary/merge_all




