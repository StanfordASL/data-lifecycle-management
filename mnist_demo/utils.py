
import numpy as np
import torch
import matplotlib.pyplot as plt
import scod
from scripts.benchmark_functions import output
from scripts.utils import eval_scod

## DEFINE VISUALIZATION FUNCTIONS
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.1307
    std = 0.3081
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp[:,:,0], cmap='Greys')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %d' % target
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if unc_model is not None:
        z_mean, z_var = unc_model(input.to(device).unsqueeze(0))
        pred = np.argmax(z_mean[0].detach().cpu().numpy())

        dist_layer = scod.distributions.CategoricalLogitLayer()
        unc = eval_scod(input, unc_model, dist_layer)
        unc = unc.item()
        xlabel += '\nPred: %d\nUnc: %0.3f' % (pred, unc)
    elif model is not None:
        # pred = np.argmax( model(input.unsqueeze(0))[0].detach().cpu().numpy() )
        pred = output(model.to(device), input.to(device), torch.tensor(target))
        xlabel += '\nPred: %d' % pred
    ax.set_xlabel(xlabel)