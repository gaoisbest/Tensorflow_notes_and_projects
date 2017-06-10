# Basic concepts
- TensorFlow separates definition of computations from their execution. Phase 1: assemble a graph. Phase 2: use a session to execute operations in the graph.
- For computation graph, **NODE: operators, variables and constants, EDGE: tensors**.
- The whole computation graph is consisted of **sub-graphs**, which can be executed parallelly across multiple CPUs or GPUs (`tf.device`).
Sub-graphs can **save computation**, since only run sub-graphs that the fetched data belongs.
- **Tensor** is an **n-dimensional array**. 0-d tensor: scalar, 1-d tensor: vector, 2-d tensor: matrix.
- Create the **Session**, evaluate the graph and fetch the value.
- **Placeholder** assembles the graph first without knowing the values needed for computation. Feed the values to placeholders using a **dictionary**. Both placeholder and Variable are feedable! `tf.Graph().is_feedable(variable)`
- Tricks
  - **Constants are stored in the graph definition**. Therefore, it makes loading graphs expensive when constants are big. Only use constants for primitive types.
  - `tf.Variable` is a class, but `tf.constant` is an op.
  - **Avoid lazy loading**. Use [Python attribute](http://danijar.com/structuring-your-tensorflow-models/) to ensure a function is only loaded the first time it’s called.

# Codes
- Basic
```
import tensorflow as tf
x = 2
y = 3
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
with tf.Session() as sess:
    # pass a list contains all variables that you want to fetch
    op1_res, op2_res = sess.run([op1, op2])
    print op1_res
    print op2_res
# out: 5 6
```

- Graph
```
import tensorflow as tf
a = tf.constant(7)
# get the Graph the tensor belongs
print a.graph # <tensorflow.python.framework.ops.Graph object at 0x3e8cb50>
assert a.graph is tf.get_default_graph()
print tf.get_default_graph() # <tensorflow.python.framework.ops.Graph object at 0x3e8cb50>

# new a user created graph
graph_example = tf.Graph()
# start the session using the new graph
session_example = tf.Session(graph = graph_example)

# get the operation name
print a.op.name # Const

# list all the operations in graph
tf.get_default_graph().get_operations() # [<tf.Operation 'Const' type=Const>]
tf.get_default_graph().version # 1
```

- Session
```
# start a new interactive session, which makes itself as the default session
interactive_session_example = tf.InteractiveSession()
interactive_tensor_example = tf.constant([7.0, 8.0, 9.0, 10.0], shape=[2, 2])

# Since the session is interactive, we can call eval() directly
print interactive_tensor_example.eval()
# out: [[  7.   8.]
        [  9.  10.]]
        
# The dynamic shape is itself a tensor describing the shape of the original tensor
# Reference: https://blog.metaflow.fr/shapes-and-dynamic-dimensions-in-tensorflow-7b1fe79be363#.1uk3xq29p
dynamic_shape = tf.shape(interactive_tensor_example)
print dynamic_shape
# out: Tensor("Shape_1:0", shape=(2,), dtype=int32)
# The shape is (2,) because interactive_tensor_example is a 2-D tensor, so the dynamic_shape is a 1-D tensor containing size of interactive_tensor_example dimensions
```

- TensorBoard
```
import tensorflow as tf

x1 = tf.constant(2, name='x1')
x2 = tf.constant(3, name='x2')

y = tf.add(x1, x2, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print sess.run(y)
writer.close()

> tensorboard --logdir='./graphs' --port 6606
```

- tf.constant is stored in the definition of graph
```
import tensorflow as tf
my_const = tf.constant([1.0, 2.0], name="my_const")
with tf.Session() as sess:
    print sess.graph.as_graph_def()

# outputs:
node {
  name: "my_const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
versions {
  producer: 21
}

```
- Variables initialization
```
# [1] initializing all variables at once
init = tf.global_variables_initializer()
sess.run(init)

# [2] Initialize only a subset of variables
init_ab = tf.variables_initializer([a, b], name="init_ab")
sess.run(init_ab)

# [3] Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
sess.run(W.initializer)
```

- Variable operation
```
# evaluate the value using eval()
W = tf.Variable(tf.truncated_normal([2, 2], name='weights'))
with tf.Session() as sess:
    sess.run(W.initializer)
    print W.eval() # equals to sess.run(W)
[[-0.42169872  1.10198021]
 [-0.09198709  0.51444685]]

a = tf.Variable(10, name='a')
assign_a = a.assign(100)
with tf.Session() as sess:
    # call assign_a directly instead of calling a.initializer, since assign() will initialize a first
    sess.run(assign_a)
    # In fact, initializer op is the assign op that assigns the variable’s initial value to the variable itself
    print sess.run(a) # 100

# assign_add() and assign_sub() can’t initialize the variable tmp_var because these ops need the original value of tmp_var
tmp_var = tf.Variable(10)
with tf.Session() as sess:
    sess.run(tmp_var.initializer) # important
    print sess.run(tmp_var.assign_add(10)) # 20
    print sess.run(tmp_var.assign_sub(2)) # 18

# use a variable to initialize another variable
# want to declare U = W * 2
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W.intialized_value())
# ensure that W is initialized before its value is used to initialize U

```

- placeholder
```
# tf.placeholder(dtype, shape=None, name=None)
# dtype is required
# shape=None means that any shape will be accepted. But you should always define the shape first.
a = tf.placeholder(tf.float32, shape=[3], name='place_a')
b = tf.constant([2.0,3.0,4.0], name='b')
c = tf.add(a, b)
with tf.Session() as sess:    
    print sess.run(c, feed_dict={a:[3.0, 2.0, 1.0]})
#out: [ 5.  5.  5.]
```

