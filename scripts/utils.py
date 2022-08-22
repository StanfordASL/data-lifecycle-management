from distutils.log import error
import numpy as np
import json
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from datetime import datetime
import scod

def set_seeds(seed=11):
    random.seed(seed)
    torch.manual_seed(seed)

class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)

class ExoRomperCamera:
    """" Utility class for accessing camera parameters. """

    # nu = 1920  # number of horizontal[pixels]
    # nv = 1200  # number of vertical[pixels]
    fpx = 1489.498779296875  # horizontal focal length[pixels]
    fpy = 1494.5677490234375  # vertical focal length[pixels]
    cx = 981.49407958984375
    cy = 558.21331787109375
    s = 0
    k = [[fpx,   s, cx],
         [0,   fpy, cy],
         [0,     0,      1]]
    K = np.array(k)


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def quatxyzw2dcm(q):

    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q0 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[1, 0] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[2, 0] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[0, 1] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[2, 1] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[0, 2] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[1, 2] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y

def sim_dead_pixels_torch(img, percent):
    assert(img.size(0)==3)
    print("WARNING: Seed not set.")
    num_pix = img.size(1)*img.size(2)
    pix_to_change = torch.randint(num_pix, (int(percent*num_pix),))
    new_img = img.clone().detach()
    for p in pix_to_change:
        a, b = divmod(p.numpy(), img.size(1))
        new_img[:, a, b] = torch.tensor([1, 0, 0])
    return new_img

def sim_dead_pixels_numpy(img, percent):
    assert(img.shape[2] == 3)
    print("WARNING: Seed not set.")
    num_pix = img.shape[0]*img.shape[1]
    pix_to_change = np.random.randint(0, num_pix, (int(percent*num_pix),))
    new_img = img.copy()
    for p in pix_to_change:
        a, b = divmod(p, img.shape[0])
        new_img[b, a, :] = np.array([255, 255, 255])
    return new_img

def sim_dead_pixels(img, percent):
    if type(img)==Image.Image:
        error('PIL image passed in.')
    elif type(img)==np.ndarray:
        return sim_dead_pixels_numpy(img, percent)
    elif type(img)==torch.Tensor:
        return sim_dead_pixels_torch(img, percent)
    else:
        error("Unknown type of image: ",type(img))

"""Plot images by passing in either a sequence of tensors or a list of filenames."""
def plot_images(imgs):
    if type(imgs) == torch.Tensor: # Sequence of tensors
        len_seq = np.shape(imgs)[0]
        fig, axs = plt.subplots(nrows=1, ncols=len_seq, figsize=(20,20*len_seq))
        for i in range(len_seq):
            img_single = imgs[i,:]
            axs[i].imshow(np.transpose(img_single, axes=[1,2,0]))
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        return fig
    elif type(imgs) == list and type(imgs[0]) == str: # List of filenames
        len_seq = len(imgs)
        fig, axs = plt.subplots(nrows=1, ncols=len_seq, figsize=(20,20*len_seq))
        for i in range(len_seq):
            axs[i].imshow(Image.open(imgs[i]).convert('RGB'))
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        return fig
    elif type(imgs) == list and type(imgs[0]) == list and type(imgs[0][0]) == str: # List of list of filenames
        n_batches = len(imgs)
        batch_size = len(imgs[0])
        fig, axs = plt.subplots(nrows=n_batches, ncols=batch_size, figsize=(20*batch_size,20*n_batches))
        for j in range(n_batches):
            for i in range(batch_size):
                axs[j,i].imshow(Image.open(imgs[j][i]).convert('RGB'))
                axs[j,i].get_xaxis().set_visible(False)
                axs[j,i].get_yaxis().set_visible(False)
        return fig
    else:
        raise ValueError("type of imgs ", type(imgs)," not supported")

## Plotting utils
def aggregate_randoms(costs_mean_from_file, accs_mean_from_file):
    rand_keys = [k for k in costs_mean_from_file.keys() if k.startswith('rand')]
    seeds = np.unique([rk.rsplit('_')[-1] for rk in rand_keys])
    flag_limits = np.sort(np.unique([int(rk.rsplit('_')[-2]) for rk in rand_keys]))
    costs_with_random = {}
    accs_with_random = {}
    # Copy over naive costs/accs
    for k in ['naive_false', 'naive_true']:
        if k not in rand_keys:
            costs_with_random[k] = costs_mean_from_file[k]
            accs_with_random[k] = accs_mean_from_file[k]
    # Aggregate random costs/accs
    for fl in flag_limits:
        fl = str(fl)
        costs_with_random['agg_rand_'+fl] = np.mean([costs_mean_from_file[rk] for rk in ['rand_'+fl+'_'+seed for seed in seeds]])
        accs_with_random['agg_rand_'+fl] = np.mean([accs_mean_from_file[rk] for rk in ['rand_'+fl+'_'+seed for seed in seeds]])
    # Copy other costs/accs
    for k in costs_mean_from_file.keys():
        if k not in rand_keys and k not in ['naive_false', 'naive_true']:
            costs_with_random[k] = costs_mean_from_file[k]
            accs_with_random[k] = accs_mean_from_file[k]
    return costs_with_random, accs_with_random

def preprocess_costs_accs(costs_mean_from_file, accs_mean_from_file, accs_std_errors_from_file):
    costs_new = {}
    accs_new = {}
    acc_std_errors_new = {}
    cost_min = costs_mean_from_file['naive_false']
    cost_max = costs_mean_from_file['naive_true']
    des_cost_min = 0
    des_cost_max = 100
    acc_min = accs_mean_from_file['naive_false']
    acc_max = accs_mean_from_file['naive_true']
    des_acc_min = 0
    des_acc_max = 100
    for k in costs_mean_from_file:
        # costs_new[k] = (costs_mean_from_file[k]-cost_min)/(cost_max - cost_min) * (des_cost_max-des_cost_min)  + des_cost_min
        costs_new[k] = (costs_mean_from_file[k])
        accs_new[k] = (accs_mean_from_file[k] - acc_min)/(acc_max - acc_min) * (des_acc_max-des_acc_min) + des_acc_min
        acc_std_errors_new[k] = accs_std_errors_from_file[k]/(acc_max - acc_min) * (des_acc_max-des_acc_min)
    return costs_new, accs_new, acc_std_errors_new

class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def visualize(self, i, partition='train', ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            q, r = self.get_pose(i)
            xa, ya = project(q, r)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
            ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
            ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')
            
            # # TEMP CODE
            # q_fake = [q[0] +0.1, q[1]-0.1, q[2]+0.1, q[3]+0.1]
            # r_fake = [1.1*r[0], r[1], 1.1*r[2]]
            # xa, ya = project(q_fake, r_fake)
            # ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=10, color='r')
            # ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=10, color='g')
            # ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=10, color='b')

        return


def init_scod(model, dataloader_to_process):
    # here, we interpret the output of the DNN as the mean of a Gaussian
    dist_constructor = lambda theta: scod.distributions.Normal(loc=theta, scale=1.)
    unc_model = scod.SCOD(model, dist_constructor, args={
        'num_eigs': 2,
        'sketch_type': 'srft',
    }, parameters=list(model.parameters())[-4:])
    unc_model.process_dataloader(dataloader_to_process)
    return unc_model

def eval_scod(input, unc_model):
    yhats, sigs = unc_model(torch.unsqueeze(input,0).cuda())
    unc = sigs.cpu().detach().numpy()
    # print("Uncertainty value:", unc)
    return unc

def sample_pred_scod(dnames, dataloaders, model, unc_model):
    true_labels = {}
    pred_labels = {}
    pred_mean = {}
    pred_stddev = {}
    uncertainties = {}
    for dn in dnames:
        print("Sampling from dataset: ",dn)
        dl = dataloaders[dn]
        imgs, lbls, fnames = next(iter(dl))
        outputs = model(imgs.cuda())
        pred_labels[dn] = outputs.cpu().detach().numpy()
        yhats, sigs = unc_model(imgs.cuda())
        true_labels[dn] = lbls.cpu().detach().numpy()
        pred_mean[dn] = np.array([yhat.loc.detach().cpu().numpy() for yhat in yhats])
        pred_stddev[dn] = np.array([yhat.scale.detach().cpu().numpy() for yhat in yhats])
        uncertainties[dn] = sigs.cpu().detach().numpy()
    # Save to file
    filename1 = 'saved_data/'+datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savez(filename1, true_labels=true_labels, pred_labels=pred_labels, pred_mean=pred_mean, pred_stddev=pred_stddev, uncertainties=uncertainties)
    print("Saved to file: ",filename1)

class PyTorchSatellitePoseEstimationDataset(Dataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='', transform=None, debug_mode = False):


        if split not in {'train', 'test', 'real_test'}:
            raise ValueError('Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

        with open(os.path.join(speed_root, split + '.json'), 'r') as f:
            label_list = json.load(f)
        
        if debug_mode:
            label_list = label_list[:1000]

        self.sample_ids = [label['filename'] for label in label_list]
        self.train = split == 'train'

        if self.train:
            self.labels = {label['filename']: {'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true']}
                            for label in label_list}
        self.image_root = os.path.join(speed_root, 'images', split)

        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        if self.train:
            q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
            y = np.concatenate([q, r])
        else:
            y = sample_id

        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image

        return torch_image, y, img_name

# NOISY VERSION
class PyTorchSatellitePoseEstimationDatasetNoisy(PyTorchSatellitePoseEstimationDataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='', transform=None, debug_mode = False, dead_percent = 0.9):
        super().__init__(split, speed_root, transform, debug_mode)
        self.dead_percent = dead_percent

    def __getitem__(self, idx):
        torch_image, y, fname = super().__getitem__(idx)
        deadened_image = sim_dead_pixels(torch_image, self.dead_percent)
        return deadened_image, y, fname

class PyTorchExoRomperDataset(Dataset):

    """ ExoRomper dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='all', root='', transform=None, debug_mode = False):

        if split not in {'all', 'space', 'earth', 'lens_flare'}:
            raise ValueError('Invalid split, has to be \'all\', \'space\', \'earth\', or \'lens_flare\' ')
        
        self.image_root = root+'images_and_labels/'
        if split in {'all'}:
            prefixes = [os.path.splitext(f)[0] for f in os.listdir(self.image_root) if f.endswith('.json') 
                            and not(f.endswith("_camera_settings.json")) 
                            and not(f.endswith("_object_settings.json"))
                            and not(f.endswith("earth.json"))
                            and not(f.endswith("space.json"))
                            and not(f.endswith("lens_flare.json"))]
        elif split in {'space', 'earth', 'lens_flare'}:
            with open(os.path.join(root, split + '.json'), 'r') as f:
                sublist_json = json.load(f)
                prefixes = [i for i in sublist_json[0]['file_prefixes']]

        if debug_mode:
            prefixes = prefixes[:10]

        self.sample_ids = list(range(len(prefixes)))

        self.labels = {}
        for index, pref in enumerate(prefixes):
            with open(os.path.join(self.image_root, pref+'.json'), 'r') as f:
                f_json = json.load(f)
                img_filename = pref+'.png'
                self.labels[img_filename] = {'q': f_json['objects'][1]['quaternion_xyzw'], 'r': f_json['objects'][1]['location'], 'pose_mat':f_json['objects'][1]['pose_transform']} 
                self.sample_ids[index] = img_filename
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, sample_id)

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
        # y = np.concatenate([q, r])
        y = np.array(r)

        if self.transform is not None:
            torch_image = self.transform(pil_image)
        else:
            torch_image = pil_image

        return torch_image, y, img_name
    
    def get_image(self, i=0):

        """ Loading image as PIL image. """
        sample_id = self.sample_ids[i]
        img_name = os.path.join(self.image_root, sample_id)
        image = Image.open(img_name).convert('RGB')
        return image
    
    def get_pose(self, i=0):

        """ Getting pose label for image. """

        sample_id = self.sample_ids[i]
        q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
        return q, r

    def visualize(self, i, ax=None):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        q, r = self.get_pose(i)
        xa, ya = self.project(q, r)   

        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')

        return
    
    def project(self, q, r, arrowlen=10):
        """ Projecting points to image frame to draw axes """
        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [arrowlen, 0, 0, 1],
                           [0, arrowlen, 0, 1],
                           [0, 0, arrowlen, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.vstack((np.hstack((quatxyzw2dcm(q), np.expand_dims(r, 1))), np.array([0,0,0,1])))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = (p_cam / p_cam[2])[0:3,:] # divide by the z coordinate

        # projection to image plane
        points_image_plane = ExoRomperCamera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y
    
    def plot_arrows(self, q=None, r=None, ax=None, i=-1):
        if q is None:
            q, r_true = self.get_pose(i)
        xa, ya = self.project(q, r, arrowlen=15)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=60, color='r', linestyle='dotted')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=60, color='g', linestyle='dotted')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=60, color='b', linestyle='dotted')

        return
    
    
