'''
代码参考：https://github.com/danielegrattarola/keras-gat/blob/master/keras_gat/graph_attention_layer.py
'''
from __future__ import absolute_import

from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
import tensorflow as tf


class GraphAttention(Layer):

    def __init__(self,
                 F_,
                 X,
                 A,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 dropout_rate=0.5,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 attn_kernel_constraint=None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.F_ = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.dropout_rate = dropout_rate  # Internal dropout rate
        self.activation = activations.get(activation)  # Eq. 4 in the paper
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)
        self.supports_masking = False

        # Populated by build()
        self.kernels = []       # Layer kernels for attention heads
        self.biases = []        # Layer biases for attention heads
        self.attn_kernels = []  # Attention kernels for attention heads
        self.X = X
        self.A = A

        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.F_ * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.F_

        super(GraphAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # assert len(input_shape) >= 2
        # F = input_shape[0][-1]

        # Initialize weights for each attention head
        for head in range(self.attn_heads):
            # # Layer kernel
            # kernel = self.add_weight(shape=(F, self.F_),
            #                          initializer=self.kernel_initializer,
            #                          regularizer=self.kernel_regularizer,
            #                          constraint=self.kernel_constraint,
            #                          name='kernel_{}'.format(head))
            # self.kernels.append(kernel)

            # # Layer bias
            if self.use_bias:
                bias = self.add_weight(shape=(self.F_, ),
                                       initializer=self.bias_initializer,
                                       regularizer=self.bias_regularizer,
                                       constraint=self.bias_constraint,
                                       name='bias_{}'.format(head))
                self.biases.append(bias)

            # Attention kernels
            attn_kernel_self = self.add_weight(shape=(self.F_, 1),
                                               initializer=self.attn_kernel_initializer,
                                               regularizer=self.attn_kernel_regularizer,
                                               constraint=self.attn_kernel_constraint,
                                               name='attn_kernel_self_{}'.format(head),)
            attn_kernel_neighs = self.add_weight(shape=(self.F_, 1),
                                                 initializer=self.attn_kernel_initializer,
                                                 regularizer=self.attn_kernel_regularizer,
                                                 constraint=self.attn_kernel_constraint,
                                                 name='attn_kernel_neigh_{}'.format(head))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])
        self.built = True

    def call(self, inputs):
        # X = inputs[0]  # total Node features (N x F_)
        # A = inputs[1]  # Adjacency matrix (N x N)
        node = inputs[0] # serial node one*hot (bs, ts, N)
        X = tf.convert_to_tensor(self.X, dtype = tf.float32)
        A = tf.convert_to_tensor(self.A, dtype = tf.float32)
        outputs_final = [] 
        for i in range(node.shape[1]):
            outputs = []
            features = K.dot(node[:,i,:], X) # (bs, F_), H
            neigh_nodes = K.dot(node[:,i,:], A) # (bs, N) 
            for head in range(self.attn_heads):
                # kernel = self.kernels[head]  # W in the paper (F x F')
                attention_kernel = self.attn_kernels[head]  # Attention kernel a in the paper (2F' x 1)

                # Compute inputs to attention network
                # features = K.dot(X, kernel)  # (N x F')

                # neigh_features = K.dot(neigh_nodes, X) # (bs, F_)

                # Compute feature combinations
                # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_j]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
                attn_for_self = K.dot(features, attention_kernel[0])    # (bs x 1), [a_1]^T [Wh_i]
                attn_for_neighs = K.dot(X, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]

                # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
                dense = attn_for_self + K.transpose(attn_for_neighs)  # (bs x N) via broadcasting

                # Add nonlinearty
                dense = LeakyReLU(alpha=0.2)(dense)

                # Mask values before activation (Vaswani et al., 2017)
                dense = tf.multiply(neigh_nodes, dense)
                mask = -10e9 * (1.0 - neigh_nodes)
                dense += mask

                # Apply softmax to get attention coefficients
                dense = K.softmax(dense)  # (bs x N)

                # Apply dropout to features and attention coefficients
                dropout_attn = Dropout(self.dropout_rate)(dense)  # (bs x N)
                dropout_attn = tf.multiply(neigh_nodes, dropout_attn)
                dropout_feat = Dropout(self.dropout_rate)(X)  # (N x F')

                # Linear combination with neighbors' features
                node_features = K.dot(dropout_attn, dropout_feat)  # (bs x F')

                if self.use_bias:
                    node_features = K.bias_add(node_features, self.biases[head])

                # Add output of attention head to final output
                outputs.append(node_features)

            # Aggregate the heads' output according to the reduction method
            if self.attn_heads_reduction == 'concat':
                output = K.concatenate(outputs)  # (bs x KF')
            else:
                output = K.mean(K.stack(outputs), axis=0)  # bs x F')

            output = self.activation(output)

            outputs_final.append(tf.expand_dims(output,axis=1))

        output_res = K.concatenate(outputs_final, axis = 1)

        return output_res

    def compute_output_shape(self, input_shape):
        # print(input_shape[0])
        output_shape = input_shape[0],input_shape[1], self.output_dim
        return output_shape
