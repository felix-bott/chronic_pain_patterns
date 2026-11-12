import torch
from torch import nn
import numpy as np


from skorch.callbacks import LRScheduler, BatchScoring, EpochScoring

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet, Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.models.modules import Expression
from braindecode import EEGRegressor

from scipy.stats import pearsonr

def pearson_corr(net, X, y):
    y_pred = net.predict(X)
    y_true = np.ravel(y)
    y_pred = np.ravel(y_pred)
    return pearsonr(y_true, y_pred)[0] 

class DataScaler(object):
    """On call multiply x with scaling_factor."""
    def __init__(self, scaling_factor):
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        return x * self.scaling_factor


def squeeze_to_ch_x_classes(x):
    """Squeeze the model output from any dimension to batch_size x n_classes."""
    while x.size()[-1] == 1 and x.ndim > 2:
        x = x.squeeze(x.ndim-1)
    return x


def create_model(model_name, window_size, n_channels, cropped, seed):
    """Create a braindecode model (either ShallowFBCSPNet or Deep4Net).

    Parameters
    ----------
    model_name: str
        The name of the model (either 'shallow' or 'deep').
    window_size: int
        The length of the input data time series in samples.
    n_channels: int
        The number of input data channels.
    cropped: bool
        A flag to switch between cropped and trialwise decoding.
    seed: int
        The seed to be used to initialize the network.

    Returns
    -------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.
    """
    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    final_conv_length = 'auto'
    if model_name == 'shallow':
        if cropped:
            window_size = None
            final_conv_length = 35
        model = ShallowFBCSPNet(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length=final_conv_length,
        )
        lr = 0.0625 * 0.01
        weight_decay = 0
    elif model_name == 'deep':
        if cropped:
            window_size = None
            final_conv_length = 1
        model = Deep4Net(
            in_chans=n_channels,
            n_classes=1,
            input_window_samples=window_size,
            final_conv_length=final_conv_length,
            stride_before_pool=True,
        )
        lr = 1 * 0.01
        weight_decay = 0.5 * 0.001
    else:
        raise ValueError(f'Model {model_name} unknown.')

    if cropped:
        to_dense_prediction_model(model)

    # remove the softmax layer from models
    new_model = torch.nn.Sequential()
    for name, module_ in model.named_children():
        if "softmax" in name:
            continue
        new_model.add_module(name, module_)

    if cropped:
        new_model.add_module('global_pool', torch.nn.AdaptiveAvgPool1d(1))
        new_model.add_module('squeeze2', Expression(squeeze_to_ch_x_classes))
    model = new_model

    # Send model to GPU
    if cuda:
        model.cuda()
        n_devices = torch.cuda.device_count()
        if n_devices > 1:
            print(f'Using {n_devices} GPUs.')
            model = nn.DataParallel(model)

    return model, lr, weight_decay


def create_estimator(
        model, n_epochs, batch_size, lr, weight_decay, n_jobs=1, train_split=None, max_epochs=None
):
    """Create am estimator (EEGRegressor) that implements fit/transform.

    Parameters
    ----------
    model: braindecode.models.Deep4Net or braindecode.models.ShallowFBCSPNet
        A braindecode convolutional neural network.
    n_epochs: int
        The number of training epochs used in model training (required to
        in the creation of a learning rate scheduler).
    batch_size: int
        The size of training batches.
    lr: float
        The learning rate to be used in network training.
    weight_decay: float
        The weight decay to be used in network training.
    n_jobs: int
        The number of workers to load data in parallel.

    Returns
    -------
    estimator: braindecode.EEGRegressor
        An estimator holding a braindecode model and implementing fit /
        transform.
    """
    # there won't be any scoring output regarding the validation set during the
    # training if used with scikit-learn functions as cross_validate as for
    # this benchmark. scikit-learn creates ids for train and validation set and
    # afterwards calls estimator.fit(train_X, train_y) without setting a
    # validation set. afterwards estimator.predict(valid_X) is executed and
    # scores are computed
    callbacks = [
        # can be dropped if there is no interest in progress of _window_ r2
        # during training
        ("R2", BatchScoring('r2', lower_is_better=False, on_train=True, name ='R2')),
        # can be dropped if there is no interest in progress of _window_ mae
        # during training
        ("MAE", BatchScoring("neg_mean_absolute_error",
                             lower_is_better=False, on_train=True, name='MAE')),
        ("PCC", BatchScoring(pearson_corr,
                             lower_is_better=False, on_train=True, name='PCC')),
        ("valid_R2", EpochScoring('r2', lower_is_better=False, on_train=False, name='valid_R2')),
        ("valid_MAE", EpochScoring("neg_mean_absolute_error",
                             lower_is_better=False, on_train=False, name='valid_MAE')),
        ("valid_PCC", EpochScoring(pearson_corr,
                             lower_is_better=False, on_train=False, name='valid_PCC')),
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Too many workers can create an IO bottleneck - n_gpus * 5 is a good
        # rule of thumb
        max_num_workers = torch.cuda.device_count() * 5
        num_workers = min(max_num_workers, n_jobs if n_jobs > 1 else 0)
    else:
        num_workers = n_jobs if n_jobs > 1 else 0

    if max_epochs is not None:
        n_epochs = max_epochs

    estimator = EEGRegressor(
        model,
        criterion=torch.nn.L1Loss,  # optimize MAE
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        max_epochs=n_epochs,
        train_split=train_split, #None,  # we do splitting via KFold object in cross_validate
        batch_size=batch_size,
        callbacks=callbacks,
        device=device,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
    )
    return estimator
