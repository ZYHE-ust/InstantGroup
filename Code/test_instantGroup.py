import torch
import argparse
from pathlib import Path
from instant_model import VAE


parser = argparse.ArgumentParser()
parser.add_argument('--test_list', default=[0, 1], type=list, help='list of testing images')
parser.add_argument('--gpu', default=0, type=int, help='GPU ID')


def load(test_list):
    base = Path(__file__).parent.parent
    img_list = [torch.load(str(base) + '/Data/mri_' + str(idx) + '.pt') for idx in test_list]

    return img_list

def test(args):

    device = 'cuda:' + str(args.gpu)

    vae = VAE()
    vae.load_state_dict(torch.load('', map_location=device))
    vae.eval()

    img_list = load(args.test_list)
    z_list = []

    with torch.no_grad():

        for x in img_list:

            recon, z = vae(x)
            z_mu, z_var = torch.chunk(z, 2, dim=1)

            z = vae.sample_z(z_mu, z_var)
            z_list.append(z)

        atlas_latent = sum(z_list) / len(z_list)

        atlas = vae.decode(atlas_latent.to(device))



if __name__ == '__main__':
    args = parser.parse_args()
    test(args)