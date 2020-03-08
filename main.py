import yaml
from model import Model


def get_config(model):
    with open('./config.yml') as f:
        model_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        return model_cfg[model]


if __name__ == '__main__':
    name = 'deepfm'
    cfg = get_config(name)
    model = Model(name, cfg)


