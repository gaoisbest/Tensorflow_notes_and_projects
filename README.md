# Tensorflow projects
- `tf_1_linear_regression.ipynb`: linear and polynomial regression.
- `tf_2_word2vec.ipynb`: word2vec. See [Chinese notes](http://url.cn/5PKmy7W), [中文解读](http://url.cn/5PKmy7W).
- `tf_3_LSTM_text_classification_version_1.ipynb`: LSTM for text classification version 1. `tf.nn.static_rnn` with single layer. See [Chinese notes](http://url.cn/5cLDOQI), [中文解读](http://url.cn/5cLDOQI).
- `tf_3_LSTM_text_classification_version_2.ipynb`: LSTM for text classification version 2. `tf.nn.dynamic_rnn` with multiple layers, variable sequence length, last relevant output. See [Chinese notes](http://url.cn/5w5VbaI), [中文解读](http://url.cn/5w5VbaI).
- `tf_4_bi-directional_LSTM_NER.ipynb`: bi-directional LSTM + CRF for brands NER. See [English notes](https://github.com/gaoisbest/NLP-Projects/blob/master/Sequence%20labeling%20-%20NER/README.md), [Chinese notes](http://url.cn/5fcC754) and [中文解读](http://url.cn/5fcC754).

# Tensorflow notes
`Lectures 1-2.md`, `Lectures 3.md` and `Lectures 4-5.md` are notes of [cs20si](http://web.stanford.edu/class/cs20si/). Each lecture includes basic concepts, codes and part solutions of corresponding assignment.

# Question and Answer
## Difference between `tf.nn.static_rnn` and `tf.nn.dynamic_rnn`.
- `static_rnn` creates an **unrolled** RNNs network by chaining cells. The weights are shared between cells. Since the network is static, the input length should be same.
- `dynamic_rnn` uses a `while_loop()`operation to run over the cell the appropriate number of times.
- Both have `sequence_length` parameter, which is a `batch_size` 1D tensor . When exceed `sequence_length`, they will **copy-through state and zero-out outputs**.

References:  
[1] Hands on machine learning with Scikit-Learn and TensorFlow P385  
[2] https://www.zhihu.com/question/52200883


