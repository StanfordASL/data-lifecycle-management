from .utils import PyTorchSatellitePoseEstimationDataset, PyTorchSatellitePoseEstimationDatasetNoisy, PyTorchExoRomperDataset, set_seeds
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, EMNIST
import shutil
import numpy as np
import datetime
import PIL
import scod

def random_number(seed = None):
    set_seeds() if seed == None else set_seeds(seed)
    return torch.rand(1)    

def refine_model(model, optimizer, inputs, labels, num_epochs, dataset_name, spec_lr = 0.001, verbose = False):
    # https://machinelearningmastery.com/update-neural-network-models-with-more-data/
    # Batch new data?
    # Ensemble model?
    _, _, _, criterion, device = set_up_model(dataset_name, spec_lr=spec_lr) # Can be reduced to not overfit
    for epoch in range(num_epochs):
        verbose and print('Epoch {}/{}'.format(epoch+1, num_epochs))
        verbose and print('-' * 10)
        # scheduler.step()
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            if criterion is None:
                dist_layer = scod.distributions.CategoricalLogitLayer()
                dist = dist_layer(outputs)
                prob_loss = -dist.log_prob(labels)
                mean_loss = prob_loss.mean()
                regularization = torch.sum(torch.stack([torch.norm(p)**2 for p in model.parameters()])) / 2e3
                loss = mean_loss + regularization
            else:
                loss = criterion(outputs, labels.float().cuda())
            loss.backward()
            optimizer.step()
        current_loss = loss.item() * inputs.size(0)
        verbose and print('Current loss {:.6f}.'.format(current_loss))
    return model



def train_model(model, scheduler, optimizer, criterion, dataloaders, device, dataset_sizes, start_epochs, num_epochs, checkpoint_path, best_model_path, losses_path, valid_loss_min_input):

    """ Training function, looping over epochs and batches. Return the trained model. """
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input
    # initialize structures to store train and valid losses
    train_loss_values = []
    valid_loss_values = []
    # epoch loop
    for epoch in range(start_epochs, start_epochs+num_epochs):
        print('Epoch {}/{}'.format(epoch+1, start_epochs+num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0

            # batch loop
            for inputs, labels, fnames in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().cuda())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                scheduler.step()
                train_loss_values.append(epoch_loss)
            else:
                valid_loss_values.append(epoch_loss)
                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss': epoch_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                # save checkpoint
                save_ckp(checkpoint, False, checkpoint_path, best_model_path)

                # save the model if validation loss has decreased
                if epoch_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_loss))
                    # save checkpoint as best model
                    save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                    valid_loss_min = epoch_loss
    # Save train losses and valid losses
    fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") +'.npz' 
    np.savez(losses_path + fname, train_loss_values = train_loss_values, valid_loss_values = valid_loss_values)

    return model

def load_losses(losses_path, fname):
    ls = np.load(losses_path + fname)
    train_loss_values = ls['train_loss_values']
    valid_loss_values = ls['valid_loss_values']
    return train_loss_values, valid_loss_values

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss = checkpoint['valid_loss']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss

def load_model_from_ckp(load_model_path):
    # Set up model
    if "ex" in load_model_path:
        dataset_name = 'exoromper'
    elif 'mnist' in load_model_path:
        dataset_name = 'mnist'
    initialized_model, scheduler, optimizer, criterion, device = set_up_model(dataset_name)
    # Load model from the saved checkpoint
    model, optimizer, start_epoch_idx, valid_loss = load_ckp(load_model_path, initialized_model, optimizer)
    return model, optimizer, start_epoch_idx, valid_loss, criterion, device

def evaluate_model(load_model_path, dataloader, position_only=True):
    # Retrieve model from ckp
    model, optimizer, start_epoch_idx, valid_loss, criterion, device = load_model_from_ckp(load_model_path)
    # Evaluate on data
    model.eval()
    if not position_only:
        q_true = np.empty((0,4), float)
        q_pred = np.empty((0,4), float)
    r_true = np.empty((0,3), float)
    r_pred = np.empty((0,3), float)
    losses = []
    for inputs, labels, fnames in dataloader:
        with torch.set_grad_enabled(False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.to(device).float()
            loss = criterion(outputs, labels.float().cuda())
        loss = loss.item()
        losses = np.append(losses, loss)

        if not position_only:
            q_labels = labels[:, :4].cpu().numpy()
            q_true = np.vstack((q_true, q_labels))
            q_batch = outputs[:, :4].cpu().numpy()
            q_pred = np.vstack((q_pred, q_batch))
        r_labels = labels[:, -3:].cpu().numpy()
        r_true = np.vstack((r_true, r_labels))
        r_batch = outputs[:, -3:].cpu().numpy()
        r_pred = np.vstack((r_pred, r_batch))
    average_loss = np.average(losses)
    if not position_only:
        return average_loss, q_true, r_true, q_pred, r_pred
    else:
        return average_loss, r_true, r_pred

def single_model_eval(load_model_path, img, position_only=False):
    if type(img) == PIL.Image.Image:
        data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        torch_img = data_transforms(img)
    elif type(img) == torch.Tensor:
        torch_img = img
    else:
        raise ValueError('img must be PIL image or Tensor, instead of type(img)')
    # Retrieve model from ckp
    model, optimizer, start_epoch_idx, valid_loss, criterion, device = load_model_from_ckp(load_model_path)
    # Evaluate on data
    model.eval()
    with torch.set_grad_enabled(False):
        new_input = torch.unsqueeze(torch_img, 0)
        input_cuda = new_input.to(device)
        outputs = model(input_cuda)
    if position_only:
        r_batch = outputs[0].cpu().numpy()
        return r_batch
    q_batch = outputs[0, :4].cpu().numpy()
    r_batch = outputs[0, -3:].cpu().numpy()
    return q_batch, r_batch    

    

def create_dataloaders(root, batch_size, dataset_name = "speed"):
    torch.manual_seed(11) # so train/test split is reproducible   
    
    if dataset_name == "speed" or dataset_name == "exoromper":
        # Processing to match pre-trained networks
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if dataset_name == "speed":
        # Loading training set, using 20% for validation, 10% for test
        full_dataset = PyTorchSatellitePoseEstimationDataset('train', root, data_transforms, debug_mode=True)
        train_and_val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * .9),
                                                                                int(len(full_dataset) * .1)])
        train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [int(len(train_and_val_dataset) * .8),
                                                                                int(len(train_and_val_dataset) * .2)])
    
        full_dataset_000 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.0)
        _, test_dataset_000 = torch.utils.data.random_split(full_dataset_000, [int(len(full_dataset_000) * .9),
                                                                                int(len(full_dataset_000) * .1)])

        full_dataset_0001 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.0001)
        _, test_dataset_0001 = torch.utils.data.random_split(full_dataset_0001, [int(len(full_dataset_0001) * .9),
                                                                                int(len(full_dataset_0001) * .1)])

        full_dataset_0005 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.0005)
        _, test_dataset_0005 = torch.utils.data.random_split(full_dataset_0005, [int(len(full_dataset_0005) * .9),
                                                                              int(len(full_dataset_0005) * .1)])

        full_dataset_001 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.001)
        _, test_dataset_001 = torch.utils.data.random_split(full_dataset_001, [int(len(full_dataset_001) * .9),
                                                                                int(len(full_dataset_001) * .1)])

        full_dataset_01 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.01)
        _, test_dataset_01 = torch.utils.data.random_split(full_dataset_01, [int(len(full_dataset_01) * .9),
                                                                                int(len(full_dataset_01) * .1)])
        full_dataset_10 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.1)
        _, test_dataset_10 = torch.utils.data.random_split(full_dataset_10, [int(len(full_dataset_10) * .9),
                                                                              int(len(full_dataset_10) * .1)])

        full_dataset_50 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.5)
        _, test_dataset_50 = torch.utils.data.random_split(full_dataset_50, [int(len(full_dataset_50) * .9),
                                                                              int(len(full_dataset_50) * .1)])

        full_dataset_90 = PyTorchSatellitePoseEstimationDatasetNoisy('train', root, data_transforms, debug_mode=True, dead_percent=0.9)
        _, test_dataset_90 = torch.utils.data.random_split(full_dataset_90, [int(len(full_dataset_90) * .9),
                                                                              int(len(full_dataset_90) * .1)])

        datasets = {'train': train_dataset, 
                    'val': val_dataset, 
                    'test': test_dataset, 
                    'test_000': test_dataset_000, 
                    'test_0001': test_dataset_0001,
                    'test_0005': test_dataset_0005, 
                    'test_001': test_dataset_001, 
                    'test_01': test_dataset_01, 
                    'test_10': test_dataset_10, 
                    'test_50': test_dataset_50, 
                    'test_90': test_dataset_90}
    elif dataset_name == "exoromper":
        full_dataset = PyTorchExoRomperDataset('all', root, data_transforms, debug_mode=False)
        train_and_val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(len(full_dataset) * .9),
                                                                                int(len(full_dataset) - int(len(full_dataset) * .9))])
        train_dataset, val_dataset = torch.utils.data.random_split(train_and_val_dataset, [int(len(train_and_val_dataset) * .8),
                                                                                int(len(train_and_val_dataset) - int(len(train_and_val_dataset) * .8))])
        space_dataset = PyTorchExoRomperDataset('space', root, data_transforms, debug_mode=False)
        earth_dataset = PyTorchExoRomperDataset('earth', root, data_transforms, debug_mode=False)
        lens_flare_dataset = PyTorchExoRomperDataset('lens_flare', root, data_transforms, debug_mode=False)
        datasets = {'all_train': train_dataset,
                    'all_val': val_dataset,
                    'all_test': test_dataset,
                    'space': space_dataset,
                    'earth': earth_dataset,
                    'lens_flare': lens_flare_dataset}
    elif dataset_name == "mnist":
        mnist_dataset = MNIST(root="~/data/",train=True,download=True,transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3801,))
                            ]))
        mnist_train, mnist_test = torch.utils.data.random_split(mnist_dataset, [int(len(mnist_dataset) * .8),
                                                                                int(len(mnist_dataset) - int(len(mnist_dataset) * .8))])
        fashion_dataset = FashionMNIST(root="~/data/",train=True,download=True,transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3801,))
                            ]))
        datasets = {'mnist_train': mnist_train,
                    'mnist_test': mnist_test,
                    'fashion': fashion_dataset}
    
    dnames = list(datasets.keys())
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in dnames}
    dataset_sizes = {x: len(datasets[x]) for x in dnames}
    return dataloaders, dataset_sizes

def set_up_model(dataset_name, spec_lr = 0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("CUDA IS AVAILABLE? ",torch.cuda.is_available())
    
    if 'exoromper' in dataset_name:
        # Getting pre-trained model and replacing the last fully connected layer
        initialized_model = models.resnet18(pretrained=True)
        num_ftrs = initialized_model.fc.in_features
        initialized_model.fc = torch.nn.Linear(num_ftrs, 3)
        initialized_model = initialized_model.to(device)  # Note: we are finetuning the model (all params trainable)

        # Setting up the learning process
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(initialized_model.parameters(), lr=spec_lr, momentum=0.9)  # all params trained
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif 'mnist' in dataset_name:
        # DNN mapping 2d input to 1d distribution parameter
        # LeNet v5
        initialized_model = nn.Sequential(
                nn.Conv2d(1, 6, 5, 1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Conv2d(6, 16, 5, 1),
                nn.ReLU(),
                nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(256, 120),
                nn.ReLU(),
                nn.Linear(120,84),
                nn.ReLU(),
                nn.Linear(84,10)
            )
        initialized_model = initialized_model.to(device)
        criterion = None # Custom loss to be specified during training
        optimizer = torch.optim.Adam(initialized_model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)

    return initialized_model, scheduler, optimizer, criterion, device

def main(num_epochs, dataloaders, dataset_sizes, resume=False, load_model_path=''):

    """ Preparing the dataset for training, setting up model, running training, and exporting submission."""

    initialized_model, exp_lr_scheduler, sgd_optimizer, criterion, device = set_up_model()
    
    print("Resuming? ", resume)

    # Training
    ckp_path = "./checkpoint/current_checkpoint.pt"
    best_path = "./best_model/best_model.pt"
    losses_path = "./losses/"
    if not resume:
        trained_model = train_model(initialized_model, exp_lr_scheduler, sgd_optimizer, criterion,
                                dataloaders, device, dataset_sizes, 0, num_epochs, ckp_path, best_path, losses_path, np.Inf)
    else:
        # load the saved checkpoint
        model, optimizer, start_epoch_idx, valid_loss = load_ckp(load_model_path, initialized_model, sgd_optimizer)

        print("start_epoch_idx = ", start_epoch_idx)
        print("valid_loss = {:.6f}".format(valid_loss))

        trained_model = train_model(model, exp_lr_scheduler, optimizer, criterion,
                                dataloaders, device, dataset_sizes, start_epoch_idx, num_epochs, ckp_path, best_path, losses_path, valid_loss)

    return ckp_path, best_path, losses_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', help='Path to the downloaded speed dataset.', default='')
    parser.add_argument('--epochs', help='Number of epochs for training.', default=20)
    parser.add_argument('--batch', help='Number of samples in a batch.', default=32)
    parser.add_argument('--resume', help='Continue training from a previous checkpoint?', dest='resume', action='store_true')
    parser.add_argument('--load_model_path', help='Path to checkpoint', default="")
    args = parser.parse_args()

    dataloaders, dataset_sizes = create_dataloaders(args.dataset, int(args.batch))
    print("Splitting datasets complete.")

    ckp_path, best_path, losses_path = main(int(args.epochs), dataloaders, dataset_sizes, resume=args.resume, load_model_path=args.load_model_path)
    print("Latest checkpoint saved at: ", ckp_path)
    print("Best model saved at: ", best_path)
    print("Losses saved at: ", losses_path)
