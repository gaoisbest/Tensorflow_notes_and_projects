# Basic concepts
- TensorFlow separates definition of computations from their execution. Phase 1: assemble a graph. Phase 2: use a session to execute operations in the graph.
- For computation graph, **NODE: operators, variables and constants, EDGE: tensors**.
- The whole computation graph is consisted of **sub-graphs**, which can be executed parallelly across multiple CPUs or GPUs (`tf.device`).
Sub-graphs can **save computation**, since only run sub-graphs that the fetched data belongs.
- **Tensor** is an **n-dimensional array**. 0-d tensor: scalar, 1-d tensor: vector, 2-d tensor: matrix.
- Create the **Session**, evaluate the graph and fetch the value.
- Tricks
  - **Constants are stored in the graph definition**. Therefore, it makes loading graphs expensive when constants are big. Only use constants for primitive types.
  - `tf.Variable` is a class, but `tf.constant` is an op.
  - tf.Variable holds several ops:
x = tf.Variable(...)
x.initializer # init op
x.value() # read op
x.assign(...) # write op
x.assign_add(...) # and more
You have to initialize your variables


# Codes
- Basic
```
import tensorflow as tf
a = tf.add(2, 3)
a # <tf.Tensor 'Add:0' shape=() dtype=int32>
with tf.Session() as sess:
    print sess.run(a) # 5
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
The easiest way is initializing all variables at once:
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
Initialize only a subset of variables:
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
sess.run(init_ab)
Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
sess.run(W.initializer)

```
