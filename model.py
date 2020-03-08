import tensorflow as tf
from hub.deepfm import DeepFM
import yaml
from pre.util import process


class Model(object):
    def __init__(self, name):
        self.cfg = get_config(name)
        self.feature_size = self.cfg['feature_size']
        self.field_size = self.cfg['field_size']
        self.embed_size = self.cfg['embed_size']

        if name == 'deepfm':
            self.epoch = self.cfg['epoch']
            self.batch = self.cfg['batch']
            self.deep_nn = self.cfg['deep_nn']
            self.dropout_fm = self.cfg['dropout_fm']
            self.dropout_deep = self.cfg['dropout_deep']
            self.output_bias = self.cfg['output_bias']

            self.model = DeepFM(self.feature_size, self.field_size, self.embed_size, self.epoch, self.batch,
                                self.deep_nn, self.dropout_fm, self.dropout_deep, self.output_bias)
        else:
            raise RuntimeError('model config error')

    def dataset(self):
        return process(self.cfg)

    def train(self, x_train, y_train, x_validate, y_validate):
        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics_list)
        self.model.fit(x_train, y_train, epochs=self.model.epoch, batch_size=self.model.batch, shuffle=True,
                       verbose=1, callbacks=[self.model.early_stopping], validation_data=(x_validate, y_validate))

    def predict(self, x_input):
        return self.model.predict(x_input)

    def save(self, name):
        self.model.save('./saved_model/' + name)


def get_config(name):
    with open('cfg.yml') as f:
        model_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        return model_cfg[name]


def load_model(name):
    return tf.keras.models.load_model('./saved_model/' + name)

