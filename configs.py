import json
import torch


class Config(object):
    def __init__(self, config, file_path="configs.json"):
        with open(file_path) as config_file:
            self._config = json.load(config_file)
            self._config = self._config.get(config)

    def get_property(self, property_name):
        return self._config.get(property_name)

    def all(self):
        return self._config


class Create(Config):
    def __init__(self):
        super().__init__('create')

    @property
    def filter_column_value(self):
        return self.get_property('filter_project')

    @property
    def slice_size(self):
        return self.get_property('slice_size')

    @property
    def joern_cli_dir(self):
        return self.get_property('joern_cli_dir')


class Data(Config):
    def __init__(self, config):
        super().__init__(config)

    @property
    def cpg(self):
        return self.get_property('cpg')

    @property
    def raw(self):
        return self.get_property('raw')

    @property
    def input(self):
        return self.get_property('input')

    @property
    def model(self):
        return self.get_property('model')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


class Paths(Data):
    def __init__(self):
        super().__init__('paths')

    @property
    def joern(self):
        return self.get_property('joern')
    @property
    def func(self):
        return self.get_property('func')
    @property
    def cpg_func(self):
        return self.get_property('cpg_func')

class Files(Data):
    def __init__(self):
        super().__init__('files')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


class Embed(Config):
    def __init__(self):
        super().__init__('embed')

    @property
    def nodes_dim(self):
        return self.get_property('nodes_dim')

    @property
    def w2v_args(self):
        return self.get_property('word2vec_args')

    @property
    def edge_type(self):
        return self.get_property('edge_type')


class TrainParas(Config):
    def __init__(self):
        super().__init__('train_paras')

    @property
    def epochs(self):
        return self.get_property('epochs')

    @property
    def early_stop_patience(self):
        return self.get_property('early_stop_patience')

    @property
    def batch_size(self):
        return self.get_property('batch_size')

    @property
    def train_test_ratio(self):
        return self.get_property('train_test_ratio')

    @property
    def shuffle(self):
        return self.get_property('shuffle')


class Tensor_GGNN_GCN(Config):
    def __init__(self):
        super().__init__('Tensor_GGNN_GCN')

    @property
    def learning_rate(self):
        return self.get_property('learning_rate')

    @property
    def weight_decay(self):
        return self.get_property('weight_decay')

    @property
    def loss_lambda(self):
        return self.get_property('loss_lambda')

    @property
    def model(self):
        return self.get_property('model')
