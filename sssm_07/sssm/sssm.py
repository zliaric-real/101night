# sssm.py
import time
from hashlib import md5
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import einops
from torch import nn
from . import saved_models

from sklearn.base import BaseEstimator, ClassifierMixin


class Model(ClassifierMixin, BaseEstimator):

    def __init__(self, device='cuda', model_name='model.pt', model_path=None, ):
        """
        if model_path is None, use default model path, you can select default model by setting model_name;
        \if you set a model_path, it should be a full path, not dir.
        :param device:
        :param model_path:
        :param model_name:
        """
        self.input_data_length = None
        self.step = None
        self.device = torch.device(device)
        self.model = self.__load_model(model_path=model_path, model_name=model_name)
        self.proba = None
        self.pred = None
        self.feature = None
        self.N_TIME = 300
        self.LABEL_SHORT = ['SS', 'KC', 'SW', 'SAW', 'VSW', 'BG', 'AR']
        self.LABEL_LONG = ['Spindle', 'K-complex', 'Slow wave', 'Sawtooth', 'Vertex Sharp', 'Background', 'Arousal']

    def __load_model(self, model_path=None, model_name='model.pt'):
        model = base_Model(Config()).to(self.device)
        if model_path is None:
            from importlib import resources
            model_path = str(resources.files(saved_models) / model_name)
        chkpoint = torch.load(model_path, map_location=self.device)
        pretrained_dict = chkpoint["model_state_dict"]
        model.load_state_dict(pretrained_dict)
        model.eval()
        return model


    def predict(self, X=None, step=None, window=20):
        """
        input data, predict sleep event for each time step
        :param step: 300 >= step >= 1
        :param X: (n_epoch, n_channel, n_time)
        :param window: the window of the median filter for predicted label
        :return: predict_proba, pred, filtered pred, feature
        """
        # check_is_fitted(self)
        # X = check_array(X)
        if len(X.shape) == 2:
            X = X[:, None, :]
        # assert X.shape[2] >= 300, 'n_time should > 300'
        # assert 300 >= step >= 1
        if step is None:
            step = X.shape[2]
        self.step = step
        self.input_data_length = X.shape[2]

        n_epoch = X.shape[0]  # n_batch = n_epoch * n_window
        sliding_window_sample = np.lib.stride_tricks.sliding_window_view(X, self.N_TIME, axis=2)[:, :, ::step]
        sliding_window_sample = einops.rearrange(sliding_window_sample,
                                                 'n_epoch n_ch n_window n_time -> (n_epoch n_window) n_ch n_time')
        sample = torch.tensor(sliding_window_sample).float().to(self.device)
        proba, fea = self.__predict(sample)
        self.feature_ = einops.rearrange(fea, '(n_epoch n_window) n_hd n_step -> n_epoch n_window n_hd n_step',
                                         n_epoch=n_epoch)
        pred = np.argmax(proba, axis=1)
        self.proba = einops.rearrange(proba, '(n_epoch n_window) n_class -> n_epoch n_window n_class',
                                      n_epoch=n_epoch)
        self.pred = einops.rearrange(pred, '(n_epoch n_window) -> n_epoch n_window', n_epoch=n_epoch)

        # return self.predict_proba, self.pred, pred_f, fea
        return self.pred.flatten()

    def predict_proba(self, data=None, step=None, window=20):
        """
        input data, predict sleep event for each time step
        :param step: 300 >= step >= 1
        :param data: (n_epoch, n_channel, n_time)
        :param window: the window of the median filter for predicted label
        :return: predict_proba, pred, filtered pred, feature
        """
        # check_is_fitted(self)
        # data = check_array(data)
        if step is None:
            step = self.step
        assert step is not None
        data = data[:, None, :]
        assert data.shape[2] >= 300, 'n_time should > 300'
        assert 300 >= step >= 1
        self.input_data_length = data.shape[2]

        n_epoch = data.shape[0]
        sliding_window_sample = np.lib.stride_tricks.sliding_window_view(data, self.N_TIME, axis=2)[:, :, ::step]
        sliding_window_sample = einops.rearrange(sliding_window_sample,
                                                 'n_epoch n_ch n_window n_time -> (n_epoch n_window) n_ch n_time')
        sample = torch.tensor(sliding_window_sample).float().to(self.device)
        proba, fea = self.__predict(sample)
        fea = einops.rearrange(fea, '(n_epoch n_window) n_hd n_step -> n_epoch n_window n_hd n_step',
                               n_epoch=n_epoch)
        pred = np.argmax(proba, axis=1)
        self.proba = einops.rearrange(proba, '(n_epoch n_window) n_class -> n_epoch n_window n_class',
                                      n_epoch=n_epoch)
        self.pred = einops.rearrange(pred, '(n_epoch n_window) -> n_epoch n_window', n_epoch=n_epoch)

        # return self.predict_proba, self.pred, pred_f, fea
        return self.proba[:, 0, :]


    def plot_predictions(self, hyp=None, epoch_ind=0, legend=True, time_unit='hours',given_ax=None,figsize=(16,3)):
        ax = self._plot_predictions(self.proba,self.LABEL_SHORT,self.N_TIME,self.step,
                               hyp=hyp, epoch_ind=epoch_ind, legend=legend, time_unit=time_unit,given_ax=given_ax,figsize=figsize)
        return ax

    @staticmethod
    def _plot_predictions(proba,LABEL_SHORT,N_TIME,step, hyp=None, epoch_ind=0, legend=True, time_unit='hours',given_ax=None,figsize=(16,3)):
        import matplotlib
        proba = pd.DataFrame(proba[epoch_ind] * 100, columns=LABEL_SHORT)
        N_WIN_PER_EPOCH = int((3000 - N_TIME) / step + 1)
        scale = {'hours':3600,'s':1}[time_unit]
        n_ind = proba.index.to_numpy() / N_WIN_PER_EPOCH * 30 / scale
        proba.set_index(n_ind, inplace=True)
        proba_df = proba[LABEL_SHORT]

        c = sns.palettes.color_palette('hsv', 20)

        paletee = [matplotlib.colors.to_hex(c[i]) for i in [0, 3, 5, 8, 10, 12, 19]]
        paletee = matplotlib.colors.ListedColormap(paletee)

        ax = proba_df.plot(kind='area', figsize=figsize, alpha=0.9, stacked=True, lw=0, sharex=False,ax=given_ax,
                           cmap=paletee)
        ax.set_ylim([0, 100])
        ax.set_xlim([n_ind[0], n_ind[-1]])
        ax.set_ylabel('Sleep event \n predict probability / %')
        ax.set_xlabel(f'Time / {time_unit}')
        if legend is True:
            ax.legend(bbox_to_anchor=(1, 1), title='Sleep events')
        else:
            ax.get_legend().remove()

        if hyp is not None:
            ax_ = ax.twinx()
            _ = hyp.plot_hypnogram(ax=ax_, lw=2)
            ax_.set_ylabel('True stage')
            ax_.set_xlabel('Time / hours')
        return ax

    def to_pandas(self, overall_threshold=0.5, describe=False, event_threshold=None):
        label_ind = np.argwhere(self.proba > overall_threshold)
        last_ind = label_ind[0]
        starts = [last_ind]
        ends = []
        for this_ind in label_ind[1:]:
            if this_ind[1] - last_ind[1] == 1 and this_ind[2] == last_ind[2]:
                last_ind = this_ind
            else:
                ends.append(last_ind)
                starts.append(this_ind)
                last_ind = this_ind
        ends.append(label_ind[-1])

        step = self.step
        starts_ = np.array(starts)
        ends_ = np.array(ends)
        proba_ = []
        proba_full = []
        for s_, e_ in zip(starts_, ends_):
            proba_.append(self.proba[s_[0], s_[1]:e_[1] + 1, s_[2]].mean())
            proba_full.append(self.proba[s_[0], s_[1]:e_[1] + 1])
        starts = starts_[:, 1] * step + int(150 - step / 2)
        ends = ends_[:, 1] * step + int(150 + step / 2)
        starts = np.where(starts <= int(150 - step / 2), 0, starts)
        ends = np.where(ends >= self.input_data_length - int(150 + step / 2), self.input_data_length, ends)

        df = pd.DataFrame({
            'Start': starts,
            'End': ends,
            'Duration': ends - starts,
            'label': [self.LABEL_LONG[label_ind] for label_ind in ends_[:, 2]],
            'predict_proba': proba_,
            'predict_proba_full': proba_full,
            'epoch_id': ends_[:, 0],
        })

        if event_threshold is None:
            event_threshold = {
                'Spindle': 0.5,
                'Background': 0.5,
                'Arousal': 0.5,
                'K-complex': 0.5,
                'Slow wave': 0.5,
                'Vertex Sharp': 0.5,
                'Sawtooth': 0.5}
        inds = []
        for label in event_threshold:
            inds.extend(df.query(f'label=="{label}" and predict_proba<{event_threshold[label]}').index.to_list())
        df = df.drop(index=inds)

        if describe is True:
            describe = df.groupby('label').describe()['predict_proba'][['count', 'mean']]
            describe['percentage'] = 100 * describe['count'] / describe['count'].sum()
            describe.rename(columns={'mean': 'predict_proba_mean'}, inplace=True)
            describe['predict_proba_mean'] *= 100
            return df, describe
        else:
            return df

    def __predict(self, data):
        self.model.eval()
        softmax = torch.nn.Softmax(dim=1)
        torch.cuda.empty_cache()
        with torch.no_grad():
            pre, fea = self.model(data)
            proba = softmax(pre).cpu().numpy()
        torch.cuda.empty_cache()
        return proba, fea
        # pre = einops.rearrange(pre, '(n_epoch n_window) n_class -> n_epoch n_window n_class', n_epoch=n_epoch)
        # fea = einops.rearrange(fea, '(n_epoch n_window) n_hd n_step -> n_epoch n_window n_hd n_step', n_epoch=n_epoch)


    @staticmethod
    def __sliding_window_sample(X, N_TIME_=300, step=50):
        n_epoch = X.shape[0]
        sliding_window_sample = np.lib.stride_tricks.sliding_window_view(X, N_TIME_, axis=2)[:, :, ::step]
        return einops.rearrange(sliding_window_sample,
                                'n_epoch n_ch n_window n_time -> (n_epoch n_window) n_ch n_time')



class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.final_out_channels = 128
        self.num_classes = 7
        self.dropout = 0.35

        self.kernel_size = 25
        self.stride = 3
        self.features_len = 15  # 15  # 127, 44

        # training configs
        self.num_epoch = 50

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 3500

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.5
        self.jitter_ratio = 2
        self.max_seg = 12


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 128
        self.timesteps = 5


def create_id():
    m = md5(str(time.perf_counter()).encode("utf-8"))
    return m.hexdigest()


def generate_body(csv, results, score=None):
    return {
        "data": {
            "csv": '/data/local-files/?d=' + csv
        },
        "predictions": [
            {
                "result": results,
                "score": round(score, 3),
            }
        ],
    }


def generate_results(start=None, end=None, label: str = None, label_from: str = None, channel: list = [],
                     description: str = None):
    return {
        "value": {
            "start": int(start),
            "end": int(end),
            "instant": False,
            "timeserieslabels": [label]
        },
        "id": create_id()[:10],
        "from_name": "label",
        "to_name": "ts",
        "type": "timeserieslabels",
        "origin": "manual",
        "meta": {
            "channel": channel,
            "label from": label_from,
            "text": description,
        }
    }
