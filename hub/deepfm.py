import tensorflow as tf


class DeepFM(tf.keras.Model):
    def __init__(self, feature_size, field_size, embed_size, epoch, batch, deep_nn, dropout_fm, dropout_deep, output_bias):
        super(DeepFM, self).__init__()
        self.feature_size = feature_size
        self.field_size = field_size
        self.embed_size = embed_size
        self.deep_nn = deep_nn

        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep
        self.output_bias = output_bias

        # fm
        self.feature_weight = tf.keras.layers.Embedding(feature_size, 1)
        self.feature_embed = tf.keras.layers.Embedding(feature_size, embed_size)

        # dnn
        for layer in range(len(deep_nn)):
            setattr(self, 'dense_' + str(layer), tf.keras.layers.Dense(self.deep_nn[layer]))
            setattr(self, 'batchNorm_' + str(layer), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(layer), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(layer), tf.keras.layers.Dropout(self.dropout_deep))

        self.fc = tf.keras.layers.Dense(1, bias_initializer=tf.keras.initializers.Constant(output_bias),
                                        activation='sigmoid')

        # hyper
        self.epoch = epoch
        self.batch = batch
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-3)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1, patience=10,
                                                               mode='max', restore_best_weights=True)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics_list = [tf.keras.metrics.TruePositives(name='tp'),
                             tf.keras.metrics.FalsePositives(name='fp'),
                             tf.keras.metrics.TrueNegatives(name='tn'),
                             tf.keras.metrics.FalseNegatives(name='fn'),
                             tf.keras.metrics.BinaryAccuracy(name='bin_acc'),
                             tf.keras.metrics.Precision(name='precision'),
                             tf.keras.metrics.Recall(name='recall'),
                             tf.keras.metrics.AUC(name='auc')]

    def call(self, inputs, training=True):
        # inputs = [feature_idx, feature_val]
        reshaped_feature_val = tf.cast(tf.reshape(inputs[1], shape=[-1, self.field_size, 1]), tf.float32)
        # linear
        weights = self.feature_weight(inputs[0])
        linear = tf.reduce_sum(tf.multiply(weights, reshaped_feature_val), 2)

        # fm
        embeddings = self.feature_embed(inputs[0])
        second_inner = tf.multiply(embeddings, reshaped_feature_val)

        summed_features_emb = tf.reduce_sum(second_inner, 1)
        summed_features_emb_square = tf.square(summed_features_emb)

        squared_features_emb = tf.square(second_inner)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

        fm = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)

        # dnn
        y_deep = tf.reshape(embeddings, shape=[-1, self.field_size * self.embed_size])
        for layer in range(0, len(self.deep_nn)):
            y_deep = getattr(self, 'dense_' + str(layer))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(layer))(y_deep, training=training)
            y_deep = getattr(self, 'activation_' + str(layer))(y_deep)
            y_deep = getattr(self, 'dropout_' + str(layer))(y_deep, training=training)

        # concat
        concat = tf.concat([linear, fm, y_deep], axis=1)
        output = self.fc(concat)
        return output


