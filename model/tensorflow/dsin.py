import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    LSTM,
    Lambda,
    Layer,
    Dropout,
    LayerNormalization,
    BatchNormalization,
    Dense,
    ReLU,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import (
    TruncatedNormal,
    GlorotUniform,
    GlorotNormal,
    Zeros,
)

from model.tensorflow.embedding import Embedding


class BiasEncoding(Layer):
    def __init__(self, sess_num, sess_len, **kwargs):
        self.sess_num = sess_num
        self.sess_len = sess_len
        super(BiasEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = int(input_shape[-1])
        # W^K_k : (1, sess_num, 1, 1)
        self.w_k = self.add_weight(
            shape=(1, self.sess_num, 1, 1), initializer=GlorotNormal(), name="w_k"
        )
        # W^T_t : (1, 1, sess_len, 1)
        self.w_t = self.add_weight(
            shape=(1, 1, self.sess_len, 1), initializer=GlorotNormal(), name="w_t"
        )
        # W^C_c : (1, 1, 1, dim)
        self.w_c = self.add_weight(
            shape=(1, 1, 1, dim), initializer=GlorotNormal(), name="w_c"
        )
        super(BiasEncoding, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, sess_num, sess_len, dim)
        # 根据广播机制实现 Eq.2: BE(k, t, c) = w_k^K + w_t^T + w_c^C
        return inputs + self.w_k + self.w_t + self.w_c


class BilinearAttention(Layer):
    def __init__(self, **kwargs):
        super(BilinearAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [target_shape, seq_shape]
        # target_shape: (B, 1, dim)
        # seq_shape: (B, T, dim)
        dim = int(input_shape[0][-1])
        # 双线性权重矩阵 W (dim x dim)
        self.W = self.add_weight(
            shape=(dim, dim), initializer=GlorotNormal(), name="bilinear_W"
        )
        super(BilinearAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        target, seq = inputs
        # 1. 计算 seq * W => (B, T, dim)
        seq_w = tf.tensordot(seq, self.W, axes=[[-1], [0]])
        # 2. 将目标项转置 => (B, dim, 1)
        target_t = tf.transpose(target, [0, 2, 1])
        # 3. 内积计算 logits (I_k W X^I) => (B, T, 1)
        logits = tf.matmul(seq_w, target_t)
        # 4. softmax 归一化注意力权重
        scores = tf.nn.softmax(logits, axis=1)
        # 5. 加权求和 (B, 1, T) x (B, T, dim) => (B, 1, dim)
        output = tf.matmul(scores, seq, transpose_a=True)
        return output


class LocalActivationUnit(Layer):
    """The LocalActivationUnit used in DIN with which the representation of
    user interests varies adaptively given different candidate items.

      Input shape
        - A list of two 3D tensor with shape:  ``(batch_size, 1, embedding_size)`` and ``(batch_size, T, embedding_size)``

      Output shape
        - 3D tensor with shape: ``(batch_size, T, 1)``.

      Arguments
        - **hidden_units**:list of positive integer, the attention net layer number and units in each layer.

        - **activation**: Activation function to use in attention net.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix of attention net.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout in attention net.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not in attention net.

        - **seed**: A Python integer to use as random seed.

      References
        - [Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. ACM, 2018: 1059-1068.](https://arxiv.org/pdf/1706.06978.pdf)
    """

    def __init__(
        self,
        hidden_units=(64, 32),
        activation="sigmoid",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        seed=1024,
        **kwargs
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                "A `LocalActivationUnit` layer should be called "
                "on a list of 2 inputs"
            )

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d and %d, expect to be 3 dimensions"
                % (len(input_shape[0]), len(input_shape[1]))
            )

        if input_shape[0][-1] != input_shape[1][-1] or input_shape[0][1] != 1:
            raise ValueError(
                "A `LocalActivationUnit` layer requires "
                "inputs of a two inputs with shape (None,1,embedding_size) and (None,T,embedding_size)"
                "Got different shapes: %s,%s" % (input_shape[0], input_shape[1])
            )
        size = (
            4 * int(input_shape[0][-1])
            if len(self.hidden_units) == 0
            else self.hidden_units[-1]
        )
        self.kernel = self.add_weight(
            shape=(size, 1), initializer=GlorotNormal(seed=self.seed), name="kernel"
        )
        self.bias = self.add_weight(shape=(1,), initializer=Zeros(), name="bias")
        self.dnn = DNN(
            self.hidden_units,
            self.activation,
            self.l2_reg,
            self.dropout_rate,
            self.use_bn,
            seed=self.seed,
        )

        super(LocalActivationUnit, self).build(
            input_shape
        )  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = K.repeat_elements(query, keys_len, 1)

        att_input = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)

        att_out = self.dnn(att_input, training=training)

        attention_score = tf.nn.bias_add(
            tf.tensordot(att_out, self.kernel, axes=(-1, 0)), self.bias
        )

        return attention_score


class DNN(Layer):
    """The Multi Layer Percetron

    Input shape
      - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

    Output shape
      - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

    Arguments
      - **hidden_units**:list of positive integer, the layer number and units in each layer.

      - **activation**: Activation function to use.

      - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

      - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

      - **use_bn**: bool. Whether use BatchNormalization before activation or not.

      - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.

      - **seed**: A Python integer to use as random seed.
    """

    def __init__(
        self,
        hidden_units,
        activation="relu",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        output_activation=None,
        seed=1024,
        **kwargs
    ):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed

        super(DNN, self).__init__(**kwargs)

    def build(self, input_shape):
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [
            self.add_weight(
                name="kernel" + str(i),
                shape=(hidden_units[i], hidden_units[i + 1]),
                initializer=GlorotNormal(seed=self.seed),
                regularizer=l2(self.l2_reg),
                trainable=True,
            )
            for i in range(len(self.hidden_units))
        ]
        self.bias = [
            self.add_weight(
                name="bias" + str(i),
                shape=(self.hidden_units[i],),
                initializer=Zeros(),
                trainable=True,
            )
            for i in range(len(self.hidden_units))
        ]
        if self.use_bn:
            self.bn_layers = [
                BatchNormalization() for _ in range(len(self.hidden_units))
            ]

        self.dropout_layers = [
            Dropout(self.dropout_rate, seed=self.seed + i)
            for i in range(len(self.hidden_units))
        ]

        self.activation_layers = [ReLU() for _ in range(len(self.hidden_units))]

        if self.output_activation:
            self.activation_layers[-1] = ReLU()

        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(
                tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i]
            )

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            try:
                fc = self.activation_layers[i](fc, training=training)
            except (
                TypeError
            ) as e:  # TypeError: call() got an unexpected keyword argument 'training'
                print("make sure the activation function use training flag properly", e)
                fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input


class BiLSTM(Layer):
    """A multiple layer Bidirectional Residual LSTM Layer.

    Input shape
      - 3D tensor with shape ``(batch_size, timesteps, input_dim)``.

    Output shape
      - 3D tensor with shape: ``(batch_size, timesteps, units)``.

    Arguments
      - **units**: Positive integer, dimensionality of the output space.

      - **layers**:Positive integer, number of LSTM layers to stacked.

      - **res_layers**: Positive integer, number of residual connection to used in last ``res_layers``.

      - **dropout_rate**:  Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.

      - **merge_mode**: merge_mode: Mode by which outputs of the forward and backward RNNs will be combined. One of { ``'fw'`` , ``'bw'`` , ``'sum'`` , ``'mul'`` , ``'concat'`` , ``'ave'`` , ``None`` }. If None, the outputs will not be combined, they will be returned as a list.


    """

    def __init__(
        self,
        units,
        layers=2,
        res_layers=0,
        dropout_rate=0.2,
        merge_mode="ave",
        **kwargs
    ):
        if merge_mode not in ["fw", "bw", "sum", "mul", "ave", "concat", None]:
            raise ValueError(
                "Invalid merge mode. "
                "Merge mode should be one of "
                '{"fw","bw","sum", "mul", "ave", "concat", None}'
            )

        self.units = units
        self.layers = layers
        self.res_layers = res_layers
        self.dropout_rate = dropout_rate
        self.merge_mode = merge_mode

        super(BiLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions"
                % (len(input_shape))
            )
        self.fw_lstm = []
        self.bw_lstm = []
        for _ in range(self.layers):
            self.fw_lstm.append(
                LSTM(
                    self.units,
                    dropout=self.dropout_rate,
                    bias_initializer="ones",
                    return_sequences=True,
                    unroll=True,
                )
            )
            self.bw_lstm.append(
                LSTM(
                    self.units,
                    dropout=self.dropout_rate,
                    bias_initializer="ones",
                    return_sequences=True,
                    go_backwards=True,
                    unroll=True,
                )
            )

        super(BiLSTM, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, mask=None, **kwargs):
        input_fw = inputs
        input_bw = inputs
        for i in range(self.layers):
            output_fw = self.fw_lstm[i](input_fw)
            output_bw = self.bw_lstm[i](input_bw)
            output_bw = Lambda(
                lambda x: K.reverse(x, 1), mask=lambda inputs, mask: mask
            )(output_bw)

            if i >= self.layers - self.res_layers:
                output_fw += input_fw
                output_bw += input_bw
            input_fw = output_fw
            input_bw = output_bw

        output_fw = input_fw
        output_bw = input_bw

        if self.merge_mode == "fw":
            output = output_fw
        elif self.merge_mode == "bw":
            output = output_bw
        elif self.merge_mode == "concat":
            output = tf.concat([output_fw, output_bw], axis=-1)
        elif self.merge_mode == "sum":
            output = output_fw + output_bw
        elif self.merge_mode == "ave":
            output = (output_fw + output_bw) / 2
        elif self.merge_mode == "mul":
            output = output_fw * output_bw
        elif self.merge_mode is None:
            output = [output_fw, output_bw]

        return output


class Transformer(Layer):
    """Simplified version of Transformer  proposed in 《Attention is all you need》

    Input shape
      - a list of two 3D tensor with shape ``(batch_size, timesteps, input_dim)`` if ``supports_masking=True`` .
      - a list of two 4 tensors, first two tensors with shape ``(batch_size, timesteps, input_dim)``,last two tensors with shape ``(batch_size, 1)`` if ``supports_masking=False`` .


    Output shape
      - 3D tensor with shape: ``(batch_size, 1, input_dim)``  if ``output_type='mean'`` or ``output_type='sum'`` , else  ``(batch_size, timesteps, input_dim)`` .


    Arguments
          - **att_embedding_size**: int.The embedding size in multi-head self-attention network.
          - **head_num**: int.The head number in multi-head  self-attention network.
          - **dropout_rate**: float between 0 and 1. Fraction of the units to drop.
          - **use_positional_encoding**: bool. Whether or not use positional_encoding
          - **use_res**: bool. Whether or not use standard residual connections before output.
          - **use_feed_forward**: bool. Whether or not use pointwise feed foward network.
          - **use_layer_norm**: bool. Whether or not use Layer Normalization.
          - **blinding**: bool. Whether or not use blinding.
          - **seed**: A Python integer to use as random seed.
          - **supports_masking**:bool. Whether or not support masking.
          - **attention_type**: str, Type of attention, the value must be one of { ``'scaled_dot_product'`` , ``'cos'`` , ``'ln'`` , ``'additive'`` }.
          - **output_type**: ``'mean'`` , ``'sum'`` or `None`. Whether or not use average/sum pooling for output.

    References
          - [Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
    """

    def __init__(
        self,
        att_embedding_size=1,
        head_num=8,
        dropout_rate=0.0,
        use_positional_encoding=False,
        use_res=True,
        use_feed_forward=True,
        use_layer_norm=True,  # 修复 2：默认开启 layernorm
        blinding=True,
        seed=1024,
        supports_masking=False,
        attention_type="scaled_dot_product",
        output_type="mean",
        **kwargs
    ):
        if head_num <= 0:
            raise ValueError("head_num must be a int > 0")
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.num_units = att_embedding_size * head_num
        self.use_res = use_res
        self.use_feed_forward = use_feed_forward
        self.seed = seed
        self.use_positional_encoding = use_positional_encoding
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm
        self.blinding = blinding
        self.attention_type = attention_type
        self.output_type = output_type
        super(Transformer, self).__init__(**kwargs)
        self.supports_masking = supports_masking

    def build(self, input_shape):
        embedding_size = int(input_shape[0][-1])
        if self.num_units != embedding_size:
            raise ValueError(
                "att_embedding_size * head_num must equal the last dimension size of inputs,got %d * %d != %d"
                % (self.att_embedding_size, self.head_num, embedding_size)
            )
        self.seq_len_max = int(input_shape[0][-2])
        self.W_Query = self.add_weight(
            name="query",
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed),
        )
        self.W_key = self.add_weight(
            name="key",
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed + 1),
        )
        self.W_Value = self.add_weight(
            name="value",
            shape=[embedding_size, self.att_embedding_size * self.head_num],
            dtype=tf.float32,
            initializer=TruncatedNormal(seed=self.seed + 2),
        )
        if self.attention_type == "additive":
            self.b = self.add_weight(
                "b",
                shape=[self.att_embedding_size],
                dtype=tf.float32,
                initializer=GlorotUniform(seed=self.seed),
            )
            self.v = self.add_weight(
                "v",
                shape=[self.att_embedding_size],
                dtype=tf.float32,
                initializer=GlorotUniform(seed=self.seed),
            )
        elif self.attention_type == "ln":
            self.att_ln_q = LayerNormalization()
            self.att_ln_k = LayerNormalization()
        if self.use_feed_forward:
            self.fw1 = self.add_weight(
                "fw1",
                shape=[self.num_units, 4 * self.num_units],
                dtype=tf.float32,
                initializer=GlorotUniform(seed=self.seed),
            )
            self.fw2 = self.add_weight(
                "fw2",
                shape=[4 * self.num_units, self.num_units],
                dtype=tf.float32,
                initializer=GlorotUniform(seed=self.seed),
            )

        self.dropout = Dropout(self.dropout_rate, seed=self.seed)
        self.ln = LayerNormalization()
        super(Transformer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.supports_masking:
            queries, keys = inputs
            query_masks, key_masks = mask
            query_masks = tf.cast(query_masks, tf.float32)
            key_masks = tf.cast(key_masks, tf.float32)
        else:
            queries, keys, query_masks, key_masks = inputs

            query_masks = tf.sequence_mask(
                query_masks, self.seq_len_max, dtype=tf.float32
            )
            key_masks = tf.sequence_mask(key_masks, self.seq_len_max, dtype=tf.float32)
            query_masks = tf.squeeze(query_masks, axis=1)
            key_masks = tf.squeeze(key_masks, axis=1)

        if self.use_positional_encoding:
            queries = self.query_pe(queries)
            keys = self.key_pe(keys)

        Q = tf.tensordot(queries, self.W_Query, axes=(-1, 0))  # N T_q D*h
        K = tf.tensordot(keys, self.W_key, axes=(-1, 0))
        V = tf.tensordot(keys, self.W_Value, axes=(-1, 0))

        # h*N T_q D
        Q_ = tf.concat(tf.split(Q, self.head_num, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, self.head_num, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, self.head_num, axis=2), axis=0)

        if self.attention_type == "scaled_dot_product":
            # h*N T_q T_k
            outputs = tf.matmul(Q_, K_, transpose_b=True)

            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "cos":
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)

            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # h*N T_q T_k

            outputs = outputs * 20  # Scale
        elif self.attention_type == "ln":
            Q_ = self.att_ln_q(Q_)
            K_ = self.att_ln_k(K_)

            outputs = tf.matmul(Q_, K_, transpose_b=True)  # h*N T_q T_k
            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        elif self.attention_type == "additive":
            Q_reshaped = tf.expand_dims(Q_, axis=-2)
            K_reshaped = tf.expand_dims(K_, axis=-3)
            outputs = tf.tanh(tf.nn.bias_add(Q_reshaped + K_reshaped, self.b))
            outputs = tf.squeeze(
                tf.tensordot(outputs, tf.expand_dims(self.v, axis=-1), axes=[-1, 0]),
                axis=-1,
            )
        else:
            raise ValueError(
                "attention_type must be [scaled_dot_product,cos,ln,additive]"
            )

        key_masks = tf.tile(key_masks, [self.head_num, 1])

        # (h*N, T_q, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-(2**32) + 1)

        # (h*N, T_q, T_k)

        outputs = tf.where(
            tf.equal(key_masks, 1),
            outputs,
            paddings,
        )
        if self.blinding:
            try:
                outputs = tf.matrix_set_diag(
                    outputs, tf.ones_like(outputs)[:, :, 0] * (-(2**32) + 1)
                )
            except AttributeError:
                outputs = tf.compat.v1.matrix_set_diag(
                    outputs, tf.ones_like(outputs)[:, :, 0] * (-(2**32) + 1)
                )

        outputs -= tf.reduce_max(outputs, axis=-1, keepdims=True)
        outputs = tf.nn.softmax(outputs)
        query_masks = tf.tile(query_masks, [self.head_num, 1])  # (h*N, T_q)
        # (h*N, T_q, T_k)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )

        outputs *= query_masks

        outputs = self.dropout(outputs, training=training)
        # Weighted sum
        # ( h*N, T_q, C/h)
        result = tf.matmul(outputs, V_)
        result = tf.concat(tf.split(result, self.head_num, axis=0), axis=2)

        if self.use_res:
            # tf.tensordot(queries, self.W_Res, axes=(-1, 0))
            result += queries
        if self.use_layer_norm:
            result = self.ln(result)

        if self.use_feed_forward:
            fw1 = tf.nn.relu(tf.tensordot(result, self.fw1, axes=[-1, 0]))
            fw1 = self.dropout(fw1, training=training)
            fw2 = tf.tensordot(fw1, self.fw2, axes=[-1, 0])
            if self.use_res:
                result += fw2
            if self.use_layer_norm:
                result = self.ln(result)

        if self.output_type == "mean":
            return tf.reduce_mean(result, axis=1, keepdims=True)
        elif self.output_type == "sum":
            return tf.reduce_sum(result, axis=1, keepdims=True)
        else:
            return result


class DSIN(tf.keras.Model):
    """
    Deep Session Interest Network (DSIN) 模型
    """

    def __init__(
        self,
        num_sparse_embs,
        dim_input_dense,
        bias,
        sess_num=3,
        sess_len=13,
        dim=128,
        head_num=8,
        dnn_hidden_units=(256, 64),
        **kwargs
    ):
        super(DSIN, self).__init__(**kwargs)
        self.sess_num = sess_num
        self.sess_len = sess_len
        self.dim = dim

        self.embedding = Embedding(num_sparse_embs, dim, dim_input_dense, bias)
        # 修复 1：Bias Encoding
        self.bias_encoding = BiasEncoding(sess_num, sess_len)

        # 1. Session Interest Extractor Layer (Transformer)
        # 用于提取单个 session 内的行为偏好
        att_embedding_size = dim // head_num
        self.transformer = Transformer(
            att_embedding_size=att_embedding_size,
            head_num=head_num,
            use_positional_encoding=False,  # Bias Encoding
            blinding=False,
            supports_masking=False,
            output_type="mean",  # Pooling 聚合为一个向量
        )

        # 2. Session Interest Interacting Layer (BiLSTM)
        # 捕捉多个 session 之间的演化关系
        self.bilstm = BiLSTM(units=dim // 2, layers=1, merge_mode="concat")

        # 3. Session Interest Activating Layer
        # 修复 3：符合公式的 BilinearAttention 层
        self.sess_att_layer = BilinearAttention()
        self.lstm_att_layer = BilinearAttention()

        # 4. MLP Layer
        self.dnn = DNN(hidden_units=dnn_hidden_units, activation="relu", use_bn=True)

        # 5. Output Layer (Logits)
        self.dense = Dense(1, activation=None, name="logits")

    def call(self, inputs, training=None):
        """
        前向传播
        inputs: 包含两个张量的列表或元组 [seq_input, target_input]
            - seq_input: (batch, seq_len, dim)，其中 seq_len = sess_num * sess_len
            - target_input: (batch, 1, dim)
        返回: (batch, 1) 的 logits
        """
        sparse_inputs, dense_inputs = inputs
        seq_input = self.embedding(sparse_inputs, dense_inputs)
        target_input = tf.reduce_mean(
            seq_input, axis=1, keepdims=True
        )  # 简单地用序列的平均作为 Target 的输入

        batch_size = tf.shape(seq_input)[0]

        # ========= 1. Session Division =========
        # 将输入序列切分为多个 session: (batch * sess_num, sess_len, dim)
        sess_input = tf.reshape(seq_input, (-1, self.sess_num, self.sess_len, self.dim))
        seq_input = self.bias_encoding(sess_input)
        sess_input = tf.reshape(seq_input, (-1, self.sess_len, self.dim))

        # 构造 Transformer 所需的 mask 长度 (全满)
        batch_sess_size = tf.shape(sess_input)[0]
        sess_lengths = tf.ones((batch_sess_size, 1), dtype=tf.int32) * self.sess_len

        # ========= 2. Session Interest Extractor =========
        # 输出: (batch * sess_num, 1, dim)
        sess_interest = self.transformer(
            [sess_input, sess_input, sess_lengths, sess_lengths], training=training
        )

        # 恢复 batch 维度，整合为 sequence of sessions: (batch, sess_num, dim)
        sess_interest = tf.reshape(sess_interest, (batch_size, self.sess_num, self.dim))

        # ========= 3. Session Interest Interacting =========
        # 输出: (batch, sess_num, dim)
        lstm_interest = self.bilstm(sess_interest, training=training)

        # ========= 4. Session Interest Activating =========
        # 计算 Target 对各 session 的注意力加权，输出均为: (batch, 1, dim)
        sess_att = self.sess_att_layer([target_input, sess_interest], training=training)
        lstm_att = self.lstm_att_layer([target_input, lstm_interest], training=training)

        # ========= 5. Feature Concat & MLP =========
        # 拼接 Target、提取兴趣、演化兴趣: (batch, 1, 3 * dim)
        concat_feat = tf.concat([target_input, sess_att, lstm_att], axis=-1)

        # 展平为二维矩阵: (batch, 3 * dim)
        concat_feat = tf.reshape(concat_feat, (batch_size, -1))

        # 经过 DNN 网络: (batch, dnn_hidden_units[-1])
        dnn_out = self.dnn(concat_feat, training=training)

        # 最终输出二维的 Logits: (batch, 1)
        logits = self.dense(dnn_out)

        return logits


if __name__ == "__main__":
    # 简单测试模型构建和前向传播
    BATCH_SIZE = 2
    NUM_SPARSE_EMBS = [
        1460,
        583,
        10131227,
        2202608,
        305,
        24,
        12517,
        633,
        3,
        93145,
        5683,
        8351593,
        3194,
        27,
        14992,
        5461306,
        10,
        5652,
        2173,
        4,
        7046547,
        18,
        15,
        286181,
        105,
        142572,
    ]
    DIM_INPUT_DENSE = 13
    DIM_INPUT_SPARSE = 26
    SESS_NUM = 3
    SESS_LEN = 13
    DIM = 128
    model = DSIN(
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_dense=DIM_INPUT_DENSE,
        bias=False,
        sess_num=SESS_NUM,
        sess_len=SESS_LEN,
        dim=DIM,
    )
    # 构造模拟输入
    sparse_inputs = tf.constant(
        np.column_stack(
            [
                np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
                for i in range(DIM_INPUT_SPARSE)
            ]
        ).astype(np.int32)
    )
    dense_inputs = tf.constant(
        np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    )
    outputs = model((sparse_inputs, dense_inputs))
    print("Logits shape:", outputs.shape)  # 应该是 (BATCH_SIZE, 1)
