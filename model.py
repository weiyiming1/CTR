from hub.deepfm import DeepFM


class Model(object):
    def __init__(self, name, cfg):
        self.feature_size = cfg['feature_size']
        self.field_size = cfg['field_size']
        self.embed_size = cfg['embed_size']

        if name == 'deepfm':
            self.epoch = cfg['epoch']
            self.batch = cfg['batch']
            self.deep_nn = cfg['deep_nn']
            self.dropout_fm = cfg['dropout_fm']
            self.dropout_deep = cfg['dropout_deep']
            self.output_bias = cfg['output_bias']

            self.model = DeepFM(self.feature_size, self.field_size, self.embed_size, self.epoch, self.batch,
                                self.deep_nn, self.dropout_fm, self.dropout_deep, self.output_bias)
        else:
            raise RuntimeError('model config error')

    def train(self, x_train, y_train, x_validate, y_validate):
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics_list)
        self.model.fit(x_train, y_train, epochs=self.model.epoch, batch_size=self.model.batch, shuffle=True,
                       verbose=1, callbacks=[self.model.early_stopping], validation_data=(x_validate, y_validate))

    def predict(self, x_input):
        return self.model.predict(x_input)

