# Basic concepts

- In TensorBoard, the solid lines represent data flow edges and the dotted arrows represent control dependence edges.

- Model should be build as a **class**. **Checkpoint**: the step at which you save your graph’s variables.

- `tf.train.Saver()`: periodically save the model’s parameters (i.e., graph’s variables) after a certain number of steps.

- `tf.name_scope` tell TensorBoard to know which nodes should be grouped together.

# Codes

- `tf.train.Saver()` default to save all variables.
```
saver = tf.train.Saver()
with tf.Session() as sess:
    for step in range(training_steps):
        sess.run([optimizer])
        if (step + 1) % 1000 == 0:
	    # model_name- + global_step
	    saver.save(sess=sess, save_path='checkpoint_directory/model_name', global_step=model.global_step) 
```

- global_step
```
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
# pass global_step as a parameter to the optimizer
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
```

- `tf.nn.embedding_lookup` looks up `ids` in a list of embedding tensors `params`.
  
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
reference:

https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do/41922877


- add summary and merge all

`tf.summary.histogram(name, values, collections=None)` outputs a summary protocol buffer with a histogram.

`tf.summary.scalar(name, tensor, collections=None)` outputs a summary protocol buffer containing a single scalar value.

`tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)` merges all summaries collected in the default graph.

# Application

[Skip-gram and Negative sampling based Word2vec](https://github.com/gaoisbest/cs20si_notes/blob/master/tf_2_word2vec.ipynb)

After running the model, execute `tensorboard --logdir=checkpoints`. To show chinese words, in left side of 'EMBEDDINGS' tab, click `Load data` button to import 'processed/vocab_200000.tsv'.


![ ](https://github.com/gaoisbest/cs20si_notes/blob/master/tf_2_word2vec.png)
