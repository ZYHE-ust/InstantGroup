import numpy as np
import random
import torch

import torch.utils.data as Data
import itertools
import glob

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






