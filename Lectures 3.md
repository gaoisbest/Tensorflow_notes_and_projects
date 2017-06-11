# Basic concepts
- For untrainable variable (e.g., `global_step` for keeping track of the number of training loop), set `trainable=False`. That is
`global_step = tf.Variable(0, trainable=False, dtype=tf.int32) increment_step = global_step.assign_add(1)`
- We can modify the gradients by using `optimizer.compute_gradients()` and `optimizer.apply_gradients()`. And using `tf.gradient()` to only train part of the network.
- [Comparisions of optimizers](http://sebastianruder.com/optimizing-gradient-descent/).
  - RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. 
  - RMSprop is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numerator update rule. 
  - Adam adds bias-correction and momentum to RMSprop.
- Higher batch size typically requires more epochs since it does fewer update steps. See [Bengio's practical tips](https://arxiv.org/pdf/1206.5533v2.pdf).
