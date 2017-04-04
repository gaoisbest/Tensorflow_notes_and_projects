

```python
import tensorflow as tf
```


```python
# tensor is an n-dimensional matrix
# 0-d tensor: scalar    
# 1-d tensor: vector    
# 2-d tensor: matrix
```


```python
# a simple example
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
with tf.Session() as sess:
    op1_res, op2_res = sess.run([op1, op2]) # pass a list contains all variables that you want to fetch
    print op1_res
    print op2_res
# out: 5 6
```


```python
tensor_example = tf.constant([2.0, 5.0, 6.0, 7.0], shape=[2, 2])
```


```python
# It is a Tensor
print tensor_example
# out: Tensor("Const:0", shape=(2, 2), dtype=float32)
```


```python
# get the Graph the tensor belongs
print tensor_example.graph
# out: <tensorflow.python.framework.ops.Graph object at 0x2b25e90>
```


```python
print tf.get_default_graph()
# out: <tensorflow.python.framework.ops.Graph object at 0x2b25e90>

# the Graph which tensor a belongs is the Default graph
assert tensor_example.graph is tf.get_default_graph()
```


```python
# get the operation name
print tensor_example.op.name
# out: Const
```


```python
# get the tensor shape
print tensor_example.get_shape()
# out: (2, 2)
print tensor_example.get_shape().as_list()
# out: [2, 2]
```


```python
# Variable
# A constant's value is stored in the graph and its value is replicated wherever the graph is loaded. 
# A variable is stored separately, and may live on a parameter server
```


```python
# tf.Variable is a class. Should initialize variables before using them
# Three ways to initialize Variables
# [1]: tf.global_variables_initializer()
a = tf.Variable([1,1], name='var_a')
b = tf.Variable([2,2], name='var_b')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print sess.run(a)
    print a.eval() # evaluate the value using eval()
# out: [1,1] [1,1]

# [2]: tf.variables_initializer()
a = tf.Variable([1,1], name='var_a')
b = tf.Variable([2,2], name='var_b')
c = tf.Variable([3,3], name='var_c')
init_ab = tf.variables_initializer([a, b], name='init_var_ab')
with tf.Session() as sess:
    sess.run(init_ab)
    print sess.run(b)
# out: [2,2]

# [3]: var.initializer
c = tf.Variable([3,3], name='var_c')
with tf.Session() as sess:
    sess.run(c.initializer)
    print sess.run(c)
# out: [3,3]
```


```python
a = tf.Variable(10, name='a')
assign_a = a.assign(100)
with tf.Session() as sess:
    sess.run(assign_a)
    print sess.run(a) # call assign_a directly instead of calling a.initializer, since assign() will initialize a first
# out: 100
```


```python
# Operations are organized in a Graph
# Nodes in graph: constants, variables and operators
# Edges in graph: tensors
# Graphs can be splitted into several chunks (sub-graph) and run them parallelly across multiple CPUs, GPUs, or devices
# Session is used to fetch the value of tensor
```


```python
# new a user created graph
graph_example = tf.Graph()

# start the session using the new graph
session_example = tf.Session(graph = graph_example)

# evaluate the tensor, but get a ERROR. Since the tensor_example is in the default graph, but the session_example is running graph_example

session_example.run(tensor_example)
# out: RuntimeError: The Session graph is empty.  Add operations to the graph before calling run().
```


```python
# add operations to graph_example (use graph.as_default()), using run() or eval() method.
with graph_example.as_default():
    tensor_with_in_graph_1 = tf.constant(8.0)
    tensor_with_in_graph_2 = tf.constant(3.0)
    tensor_sums = tf.add(tensor_with_in_graph_1, tensor_with_in_graph_2)
    print session_example.run(tensor_sums)
    print tensor_sums.eval(session = session_example)
out: 
11.0
11.0 

print graph_example
# out: <tensorflow.python.framework.ops.Graph object at 0x2607b10>
```


```python
# list all the operations in graph_example
print graph_example.get_operations()
# out: [<tf.Operation 'Const' type=Const>, <tf.Operation 'Const_1' type=Const>, <tf.Operation 'Add' type=Add>]
print graph_example.version
# out: 3

# close the session
session_example.close()
```


```python
# start a new interactive session, which makes itself as the default session
interactive_session_example = tf.InteractiveSession()

interactive_tensor_example = tf.constant([7.0, 8.0, 9.0, 10.0], shape=[2, 2])

# Since the session is interactive, we can call eval() directly
print interactive_tensor_example.eval()
# out: [[  7.   8.]
        [  9.  10.]]
```


```python
# The dynamic shape is itself a tensor describing the shape of the original tensor
# Reference: https://blog.metaflow.fr/shapes-and-dynamic-dimensions-in-tensorflow-7b1fe79be363#.1uk3xq29p
dynamic_shape = tf.shape(interactive_tensor_example)
print dynamic_shape
# out: Tensor("Shape_1:0", shape=(2,), dtype=int32)
# The shape is (2,) because interactive_tensor_example is a 2-D tensor, so the dynamic_shape is a 1-D tensor containing size of interactive_tensor_example dimensions

```


```python
placeholder
```


```python
# tf.placeholder(dtype, shape=None, name=None)
# dtype is required
# shape=None means that any shape will be accepted. But you should always define the shape first.
a = tf.placeholder(tf.float32, shape=[3], name='place_a')
b = tf.constant([2.0,3.0,4.0], name='b')
c = tf.add(a, b)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(c, feed_dict={a:[3.0, 2.0, 1.0]})
#out: [ 5.  5.  5.]
```


```python
# Tensorboard example: tensorboard_test.py
# Ops are exported to an event file. Tensorboard converts event files to graph.
a = tf.constant(2, name='a')
b = tf.constant(5, name='b')
c = tf.add(a, b, name='add_op')
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph) # write ops to event file
    print sess.run(c)
writer.close()

> python tensorboard_test.py # run the script
> tensorboard --logdir = './graphs' # start tensorboard graph
# output:
Starting TensorBoard 41 on port 6006
(You can navigate to http://x.x.x.x:6006)
...
# then type this link in browser to see results
```
