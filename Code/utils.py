import numpy as np
import random
import torch
import math
import glob
import torch.utils.data as Data
import itertools


def mse_loss(x, y):
    return torch.mean( (x - y) ** 2 )


def ncc_loss(I, J, win=None, device='cuda'):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    sum_filt = torch.ones([1, 1, *win]).to(device)

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


def gradient_loss(s, penalty='l2'):
    if len(s.shape) == 5:
        dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
        dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
        dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

        if(penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz))/3.0
    else:
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        d *= 0.5
    return d


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    ndims = len(list(I.size())) - 2

    conv_fn = getattr(F, 'conv%dd' % ndims)

    I_sum = conv_fn(I, filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross

def sample_z(mean, logvar):
    """
    Reparameterization trick, sample from a Gaussian with given mean and log-variance
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mean


def set_seed(seed=0):
    """
    Set the random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_img(idx):
    """
    Load image of index idx to torch tensor
    """
    img = None
    return img


class Dataset_mri(Data.Dataset):
    """
    Customized dataset for pairwise registration and InstantGroup models
    """
    def __init__(self, path, reference_size =[96, 112, 96], data_range = [100, 425], specific_list=None, setsize=2):
        """
        Parameters:
            path: Folder path of the data files.
            reference_size: Shape of the data.
            data_range: Range of the data for training/validation.
            specific_list: List of specified index for specific testing purposes.
            setsize: The number of images for each iteration, default is 2 for pairwise registration and training stage of InstantGroup.
        """
        super(Dataset_mri, self).__init__()
        self.path = path
        self.reference_size = reference_size
        self.setsize=setsize
        self.mri = sorted(glob.glob(self.path + "*_mri.nii.gz"))
        self.tensor = sorted(glob.glob(self.path+'*_tensor'))
        self.indice = [x for x in range(data_range[0], data_range[1])] if not specific_list else specific_list
        self.index_pair = list(itertools.permutations(self.indice, setsize))

        # print('dataset initialized')

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        output = []
        for i in range(self.setsize):
            idx = self.index_pair[step][i]
            img = load_img(idx)
            output += [img]
        return tuple(output)
