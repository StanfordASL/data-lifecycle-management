from cgi import test
import numpy as np
import torch
from matplotlib import pyplot as plt
import copy
import random
from random import randint
from torch.cuda.amp.autocast_mode import autocast
import datetime
from .train_functions import load_model_from_ckp, create_dataloaders, refine_model
from .utils import init_scod, eval_scod, plot_images, set_seeds



def benchmark_individual(OOD_flagger, model, optimizer, refine_model, loss, test_seq, labels, dataset_name, num_refine_epochs = 4, refine_lr = 0.001):
    flagged = np.zeros(len(test_seq))
    outs    = np.zeros_like(labels)
    for ind, input in enumerate(test_seq):
        input = input[None, :].cuda()
        flag = OOD_flagger(input)
        if not flag:
            outs[ind] = model(input).detach().cpu()
        else:
            print("FLAGGED at index ",ind)
            true_label = labels[ind]
            model = refine_model(model, optimizer, input, true_label, num_refine_epochs, dataset_name, spec_lr=refine_lr)
            flagged[ind] = True
            outs[ind] = model(input).detach().cpu()
    # Calculate metrics
    cost_per_label = 1
    relabel_cost = cost_per_label * np.cumsum(flagged)

    accuracy = -np.array([loss(out, labels[ind].detach().cpu().numpy()) for ind, out in enumerate(outs)])
    return relabel_cost, accuracy

def output(model, inp, label):
    current_out = model(torch.unsqueeze(inp,0)).detach().cpu()
    if not (current_out.shape == label.shape):
        current_out = np.argmax(current_out.numpy())
    return current_out

def benchmark_batch(OOD_flagger_batch, model, optimizer, refine_model, loss, test_seq, labels, dataset_name, num_refine_epochs = 300, refine_lr = 0.1, verbose = False):
    num_batches, batch_size, _, _, _ = test_seq.shape
    flagged = np.zeros((num_batches, batch_size))
    outs    = np.zeros_like(labels)
    accuracy = np.zeros(num_batches*batch_size)
    for bidx in range(num_batches):
        batch = test_seq[bidx,:,:,:,:]
        input = batch.cuda()
        flags = OOD_flagger_batch(input) # returns a flag for each img in input
        for i in range(batch_size):
            flag = flags[i]
            inp = input[i]
            true_label = labels[bidx,i]
            current_out = output(model, inp, true_label)
            if not flag:
                verbose and print("True label:", true_label)
                verbose and print("Output:", current_out)
                outs[bidx, i] = current_out
            else:
                verbose and print("True label:", true_label)
                verbose and print("FLAGGED input ", i, " in batch ", bidx)
                verbose and print("Before refinement output:", current_out)
                verbose and print("Before refinement loss:",loss(current_out, true_label.detach().cpu()))
                model = refine_model(model, optimizer, torch.unsqueeze(input[i],0), true_label, num_refine_epochs, dataset_name, spec_lr=refine_lr)
                flagged[bidx, i] = True
                outs[bidx, i] = output(model, inp, true_label)
                verbose and print("After refinement output:", outs[bidx, i])
                verbose and print("After refinement loss:",loss(torch.from_numpy(outs[bidx, i]), labels[bidx, i].detach().cpu()))
            mse_loss = loss(torch.from_numpy(outs[bidx,i]), labels[bidx, i].detach().cpu())
            accuracy[bidx*batch_size + i] = -(mse_loss.numpy())
    # Calculate metrics
    cost_per_label = 1
    relabel_cost = (cost_per_label * np.cumsum(flagged)).astype(int)

    verbose and print('accuracy:',accuracy)
    return relabel_cost, accuracy

def loss(output, true_label):
    return torch.nn.MSELoss()(output, true_label)



def create_benchmark_seq(dataset_path, dict_img_src, position_only=True):
    batch = 20
    if "speed" in dataset_path:
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch)
    elif "exoromper" in dataset_path:
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch, dataset_name='exoromper')
    
    test_seq = torch.empty((0, 3, 224, 224))
    if position_only:
        labels   = torch.empty((0,3))
    else:
        labels   = torch.empty((0,7))
    fnames = []

    for dn in dict_img_src:
        dl = dataloaders[dn] 
        if "speed" in dataset_path:
            imgs, lbls = next(iter(dl))
        elif "exoromper" in dataset_path:
            imgs, lbls, fnms = next(iter(dl))
        imgs_np = imgs
        lbls_np = lbls
        fnms = list(fnms)
        num_imgs = dict_img_src[dn]
        these_imgs = imgs_np[:num_imgs]
        these_lbls = lbls_np[:num_imgs]
        these_fnames = fnms[:num_imgs]
        print("Imgs from ",dn, " :", np.shape(these_imgs))
        test_seq = torch.cat((test_seq, these_imgs), dim=0)
        labels = torch.cat((labels, these_lbls), dim = 0)
        fnames = fnames + these_fnames
        print("Test seq so far :", np.shape(test_seq))
        print("Labels so far :", np.shape(labels))

    return test_seq, labels, fnames

def create_benchmark_seq_batches(dataset_path, batch_size, num_batches, batch_compositions, position_only=True, seed=None):
    set_seeds() if seed == None else set_seeds(seed)
    if "speed" in dataset_path:
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch_size)
    elif "ex" in dataset_path:
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch_size, dataset_name='exoromper')
    elif "mnist" in dataset_path:
        dataloaders, dataset_sizes = create_dataloaders(dataset_path, batch_size, dataset_name='mnist')
    
    if "speed" in dataset_path or "ex" in dataset_path:
        test_seq = torch.empty((num_batches, batch_size, 3, 224, 224))
        if position_only:
            labels   = torch.empty((num_batches, batch_size, 3))
        else:
            labels   = torch.empty((num_batches, batch_size, 7))
        fnames = [["" for i in range(batch_size)] for j in range(num_batches)]
    elif "mnist" in dataset_path:
        test_seq = torch.empty((num_batches, batch_size, 1, 28, 28))
        labels   = torch.empty((num_batches, batch_size, 1))
        fnames = [["" for i in range(batch_size)] for j in range(num_batches)]
    
    for b in range(num_batches):
        idx_in_batch = 0
        for dn in batch_compositions[b]:
            dset = dataloaders[dn].dataset
            for i in range(batch_compositions[b][dn]):
                if "speed" in dataset_path or "exoromper" in dataset_path:
                    img, lbl, fnm = dset.__getitem__(randint(0, len(dset)-1))
                elif "mnist" in dataset_path:
                    ri = randint(0, len(dset)-1)
                    fnm = str(ri)
                    img, lbl = dset.__getitem__(ri)
                test_seq[b,idx_in_batch,:,:,:] = img
                labels[b,idx_in_batch,:] = torch.from_numpy(np.array(lbl))
                fnames[b][idx_in_batch] = fnm
                idx_in_batch += 1
        if not(idx_in_batch == batch_size):
            raise ValueError("Sum of batch compositions %i does not equal batch_size %i"%(idx_in_batch, batch_size))

    return test_seq, labels, fnames

def alg_flags(algs_to_test, dataset_path, batch_size, num_batches, batch_compositions, position_only=True):
    # Create batches of images
    test_seq, labels, fnames = create_benchmark_seq_batches(dataset_path, batch_size, num_batches, batch_compositions, position_only=position_only)
    # For each batch, get flags from each algorithm
    num_algs = len(algs_to_test)
    flags = [[[] for j in range(num_algs)] for i in range(num_batches)]
    for i in range(num_batches):
        for (j, alg) in enumerate(algs_to_test):
            flags[i][j] = alg(test_seq[i])
            print("Test seq ",i," algorithm ",j," flagged: ",flags[i][j])
    return test_seq, labels, fnames, flags

def eval_flaggers(flaggers, load_model_path, test_seq, labels, indiv=True, refine_function=refine_model, loss_function = loss, fnprefix=""):
    costs_mean = {}
    accs_mean = {}
    accs_std = {}
    for (i,flagger) in enumerate(flaggers):
        alg, algname = flagger
        print("Evaluating algorithm ",algname)
        cost, acc = eval_flagger(alg, load_model_path, test_seq, labels, indiv=indiv, refine_function=refine_function, loss_function = loss_function)
        
        ### SAVE TO FILE
        fname = fnprefix + '_' + algname + '_cost_acc.npz'
        full_path = './saved_data/alg_specific/' + fname
        print("Saving cost/acc data for ", algname," to file ",fname)
        np.savez(full_path, cost = cost, acc = acc)
        ###

        costs_mean[algname]=np.mean(cost)
        accs_mean[algname]=np.mean(acc)
        accs_std[algname]=np.std(acc)

        ### SAVE TO FILE
        fname = fnprefix + '_cost_acc_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +'.npz' 
        full_path = './saved_data/costs_and_accs/' + fname
        print("Saving mean cost/ mean acc/ stddev cost of algorithms so far to file ",fname)
        np.savez(full_path, costs_mean = costs_mean, accs_mean = accs_mean, accs_std = accs_std)
        ###
    return costs_mean, accs_mean, accs_std

def load_costs_accs_from_file(fname):
    full_path = './saved_data/costs_and_accs/' + fname
    ls = np.load(full_path, allow_pickle=True)
    costs_mean = ls['costs_mean'][()]
    accs_mean = ls['accs_mean'][()]
    accs_std = ls['accs_std'][()]
    return costs_mean, accs_mean, accs_std

def load_costs_accs_alg_specific_from_file(fname):
    full_path = './saved_data/alg_specific/' + fname
    ls = np.load(full_path)
    cost_vec = ls['cost']
    acc_vec = ls['acc']
    return cost_vec, acc_vec

def load_categorical_loss_from_file(fname):
    full_path = './saved_data/alg_specific/' + fname
    ls = np.load(full_path)
    loss_means = ls['loss_means']
    loss_stddev = ls['loss_stddev']
    cost_totals = ls['cost_totals']
    return loss_means, loss_stddev, cost_totals

def eval_flagger(ood_flagger, load_model_path, test_seq, labels, indiv=True, refine_function=refine_model, loss_function = loss):
    # Load model afresh each time so that it is not overwritten by previous benchmark call
    model, optimizer, start_epoch_idx, valid_loss, criterion, device = load_model_from_ckp(load_model_path)
    dataset_name = determine_dataset_from_path(load_model_path)
    if indiv==True:
        cost, acc = benchmark_individual(ood_flagger, model, optimizer, refine_function, loss_function, test_seq, labels, dataset_name)
    else:
        cost, acc = benchmark_batch(ood_flagger, model, optimizer, refine_function, loss_function, test_seq, labels, dataset_name)
    return cost, acc

def determine_dataset_from_path(load_model_path):
    if "ex" in load_model_path:
        return 'exoromper'
    elif 'mnist' in load_model_path:
        return 'mnist'
    else:
        raise("Unable to determine dataset type.")

def create_scod_model(load_model_path, dataset, batch_size):
    # Load model afresh each time so that it is not overwritten by previous benchmark call
    model, optimizer, start_epoch_idx, valid_loss, criterion, device = load_model_from_ckp(load_model_path)
    dataset_name = determine_dataset_from_path(load_model_path)
    dataloaders, dataset_sizes = create_dataloaders(dataset, batch_size, dataset_name=dataset_name)
    if "ex" in dataset_name:
        unc_model = init_scod(model, dataloaders["space"], dataset_name)
    elif 'mnist' in dataset_name:
        unc_model = init_scod(model, dataloaders["mnist_train"], dataset_name)        
    return unc_model

## Functions to support batch using Jacobian diagonals
def _ell_w_old(Jzs, w):
    return np.sum([Jzs[i]*w[i] for i in range(len(w))])

def _inner(a,b):
    return np.sum(np.inner(a,b))

def batch_flagger_old(x, batch_size, unc_model, flag_limit=None, debug=False):
    if flag_limit == None:
        raise ValueError("flag_limit must be specified.")
    # Compute norms
    # Jzs = [unc_model.get_J_z_s(torch.unsqueeze(xi,0).cuda()).detach().cpu().numpy() for xi in x]
    Jzs_torch = unc_model.get_J_z_s(x.cuda())
    Jzs = [Ji.detach().cpu().numpy().diagonal() for Ji in Jzs_torch]
    debug and print("Jzs shape:", len(Jzs))
    ell = np.sum(Jzs)
    debug and print("ell shape:", ell.shape)
    sigmas = [np.sqrt(_inner(Jz, Jz)) for Jz in Jzs]
    debug and print("sigmas len:", len(sigmas))
    debug and print("sigmas:", sigmas)
    sigma_sum = np.sum(sigmas)
    debug and print("sigma_sum:", sigma_sum)
    # Frank-Wolfe optimization
    # Initialize weights to 0
    w = np.array([0.0 for xi in x])
    for b in range(flag_limit):
        # Greedily select point f
        vals = [1/sigmas[n] * np.sum([(1-w[m])*_inner(Jzs[m], Jzs[n]) for m in range(batch_size)]) for n in range(batch_size)]
        debug and print("vals:", vals)
        f = np.argmax(vals)
        debug and print("selecting f:", f)
        # Perform line search for step size gamma
        num = _inner(sigma_sum/sigmas[f] * Jzs[f] - _ell_w_old(Jzs,w), ell-_ell_w_old(Jzs,w))
        den = _inner(sigma_sum/sigmas[f] * Jzs[f] - _ell_w_old(Jzs,w), sigma_sum/sigmas[f] * Jzs[f] - _ell_w_old(Jzs,w))
        gamma = num/den 
        debug and print("selecting gamma:", gamma)
        # Update weight for newly selected point
        debug and print("np.eye(batch_size)[f]: ", np.eye(batch_size)[f])
        w = (1-gamma)*w + gamma * sigma_sum / sigmas[f] * np.eye(batch_size)[f]
        debug and print("new w:", w)
    # Result: minimize (1-w).T @ K @ (1-w)
    flags = [w[i]>0 for i in range(batch_size)]
    return flags

## Functions to support batch using Lt_J

def get_Lt_J(xi, unc_model, debug, dist_layer=None):
    with autocast():
        z = unc_model.model(xi.cuda()) # get params of output dist
        if hasattr(unc_model, 'dist_constructor'):
            dist = unc_model.dist_constructor(z)
        else:
            dist = dist_layer
        Lt_z = dist.apply_sqrt_F(z).mean(dim=0) # L^\T theta
        # flatten
        Lt_z = Lt_z.view(-1)    
    Lt_J = unc_model._get_weight_jacobian(Lt_z) # L^\T J, J = dth / dw
    debug and print("Lt_J shape: ",Lt_J.shape)
    return Lt_J.detach().cpu().numpy()

def kernel_LtJ(LtJ_1, LtJ_2, debug):
    debug and print("LtJ_1 shape: ",LtJ_1.shape)
    debug and print("LtJ_2 shape: ",LtJ_2.shape)
    kernel = np.trace(np.inner(LtJ_1, LtJ_2))
    debug and print("kernel: ",kernel)
    return kernel

def _ell_w_Ltj(ells, w):
    return np.sum([ells[i]*w[i] for i in range(len(w))], 0)

def ds_scod_flagger(x, unc_model, flag_limit=None, debug=False, dist_layer=None):
    if flag_limit == None:
        raise ValueError("flag_limit must be specified.")
    debug and print("x shape: ",x.shape)
    batch_size = x.shape[0]

    # Compute belief updates
    ell = np.stack([get_Lt_J(x[i].unsqueeze(0), unc_model, debug, dist_layer=dist_layer) for i in range(batch_size)])
    debug and print("ell shape: ",ell.shape)
    ell_sum = np.sum(ell, 0)
    debug and print("ell_sum shape: ",ell_sum.shape)

    # sigmas = [np.linalg.norm(ell[i]) for i in range(batch_size)]
    sigmas = [np.sqrt(np.linalg.norm(np.inner(ell[i],ell[i]))) for i in range(batch_size)]
    debug and print("sigmas len:", len(sigmas))
    debug and print("sigmas:", sigmas)
    sigma_sum = np.sum(sigmas)
    debug and print("sigma_sum:", sigma_sum)

    

    # Frank-Wolfe optimization
    # Initialize weights to 0
    w = np.array([0.0 for xi in x])
    for b in range(flag_limit):
        # Greedily select point f
        vals = [1/sigmas[n] * np.sum([(1-w[m])*kernel_LtJ(ell[m], ell[n], False) for m in range(batch_size)]) for n in range(batch_size)]
        debug and print("vals:", vals)
        f = np.argmax(vals)
        debug and print("selecting f:", f)
        
        # Perform line search for step size gamma
        ell_w = _ell_w_Ltj(ell,w)
        debug and print("ell_w shape: ",ell_w.shape)
        num = np.inner(sigma_sum/sigmas[f] * ell[f] - ell_w, ell_sum - ell_w)
        debug and print("num shape: ",num.shape)
        debug and print("num : ",num)
        den = np.inner(sigma_sum/sigmas[f] * ell[f] - ell_w, sigma_sum/sigmas[f] * ell[f] - ell_w)
        debug and print("den shape: ",den.shape)
        debug and print("den : ",den)
        gamma = np.trace(num)/np.trace(den) 
        debug and print("selecting gamma:", gamma)
        
        # Update weight for newly selected point
        debug and print("np.eye(batch_size)[f]: ", np.eye(batch_size)[f])
        w = (1-gamma)*w + gamma * sigma_sum / sigmas[f] * np.eye(batch_size)[f]
        debug and print("new w:", w)
    # Result: minimize (1-w).T @ K @ (1-w)
    flags = [w[i]>0 for i in range(batch_size)]
    debug and print("flagging: ",flags)
    return flags

"""Specify either threshold or a flag_limit"""
def scod_flagger(x, unc_model, thres = None, flag_limit = None, debug = False, dist_layer = None):
    if (thres == None and flag_limit == None) or (thres != None and flag_limit != None):
        raise ValueError("Exactly ONE of thres or flag_limit must be specified. Current thres: ", thres,". flag_limit: ", flag_limit)
    uncs = np.ndarray.flatten(np.array([eval_scod(xi, unc_model, dist_layer=dist_layer) for xi in x]))
    debug and print("uncs:",uncs)
    if thres != None:
        return [unc >= thres for unc in uncs]
    elif flag_limit != None:
        highest_uncs_inds = np.argpartition(uncs, -flag_limit)[-flag_limit:]
        flags = [True if i in highest_uncs_inds else False for i in range(len(uncs))]
        return flags

def scod_uncertainties(test_seq, unc_model):
    num_batches, batch_size, _, _, _ = test_seq.shape
    uncs = []
    for bidx in range(num_batches):
        batch = test_seq[bidx,:,:,:,:]
        input = batch.cuda()
        uncs = np.append(uncs,[eval_scod(xi, unc_model) for xi in input])
    return uncs

"""Picks exactly flag_limit number of inputs in x to flag as True"""
def random_flagger(x, flag_limit=None, seed=11):
    if flag_limit == None:
        raise ValueError("flag_limit must be specified.")
    np.random.seed(seed)
    inds = np.random.choice(len(x), flag_limit, replace=False)
    flags = [True if i in inds else False for i in range(len(x))]
    return flags

def plot_timeseries_loss(num_batches, loss_means, loss_stddev, cost_totals, minimal=False, legend_external=False):
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(2,1, height_ratios=[1, 1])
    ax0 = fig.add_subplot(gs[0])

    ax0.get_xaxis().set_visible(False)
    if not minimal: ax0.set_ylabel("Loss", fontsize=18)
    ax0.set_ylim([2,12]) # Set to be consistent across all categories
    ax0.grid()
    plt.yticks(fontsize=18)

    ax1 = fig.add_subplot(gs[1], sharex = ax0)
    if not minimal: ax1.set_ylabel("Cumulative Cost", fontsize=18)
    ax1.set_ylim([0, 200])
    ax1.grid(axis='y')
    if not minimal: ax1.set_xlabel("Batches", fontsize=18)
    plt.xticks(ticks=np.arange(num_batches+1), fontsize=18)
    plt.yticks(fontsize=18)

    fig.subplots_adjust(hspace=.05)

    width = 0.6/len(loss_means)
    starting_width_offset = -(len(loss_means)-1)*width/2

    # Naive false, naive true, random, scod, diverse
    # colors = ['#56638A','#483A58','#b7ab24','#d9525e','#59a590']
    colors = ['#443850','#FF7F0E','#b7ab24','#d9525e','#59a590']
    # colors = ['#586994','#04724D','#b7ab24','#d9525e','#59a590']
    labels = ['naive_false', 'naive_true', 'random','scod', 'diverse']

    ind = np.arange(1, num_batches+1)
    for i in range(len(loss_means)):
        batch_loss = loss_means[i]
        batch_stddev = loss_stddev[i]
        batch_cost = cost_totals[i]
        ax0.errorbar(ind, batch_loss, yerr=batch_stddev, ls='--',capsize=3, color=colors[i],label=labels[i], marker='o',markersize=10)
        
        ax1.bar(ind+i*width+starting_width_offset, 1.0+np.array(batch_cost), width, color=colors[i],label=labels[i])
    
    if not minimal and not legend_external: ax0.legend(fontsize=15)
    if not minimal and not legend_external: ax1.legend(fontsize=15)

    if legend_external: 
        # Shrink current axis by 20%
        box = ax0.get_position()
        ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax0.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=24, ncol=5)
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=24, ncol=5)

        
    
    return fig


def plot_another(acc_data, cost_data):
    colors = ['darkorange', 'springgreen', 'lime','limegreen','forestgreen',  'darkgreen', 'blue']

    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel("Relabeling cost")
    ax.set_xlabel("Average model performance")
    ax.grid()


    for i in range(len(acc_data)):
        ax.scatter(acc_data[i], cost_data[i], color=colors[i])
    return fig

if __name__== '__main__':
    # Requires model with label of shape 7
    # load_model_path = './best_model/images_1000_utils_best_model.pt'
    # dataset_path = 'datasets/speed/'
    # dict_img_src = {'train':5, 'test_001':5, 'test_50':5}

    dataset_path = 'datasets/exoromper/'
    dict_img_src = {'space':5, 'earth':5, 'lens_flare':5}
    test_seq, labels = create_benchmark_seq(dataset_path, dict_img_src, position_only=True)
    plot_images(test_seq)

    load_model_path = './best_model/ex_v5_best_model.pt'
    naive_false = lambda x: False
    cost_nf, acc_nf = eval_flagger(naive_false, load_model_path, test_seq, labels, indiv=True)
    naive_true = lambda x: True
    cost_nt, acc_nt = eval_flagger(naive_true, load_model_path, test_seq, labels, indiv=True)

    acc_data = [acc_nf, acc_nt]
    cost_data = [cost_nf, cost_nt]

    plot_timeseries(test_seq, acc_data, cost_data)
    plot_another(acc_data, cost_data)