import argparse
from torch.optim import Adam
from torch.cuda.amp import autocast
from tqdm import tqdm
import os
from instant_model import VAE
from reg_model import VxmDense
from utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--description', default='instantGroup', type=str, help='description for the model')
parser.add_argument('--data_dir', default='', type=str, help='training data directory')
parser.add_argument('--model_dir', default='./checkpoints/', type=str, help='model output directory')
parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--num_epoch', default=3, type=int, help='number of training epochs')
parser.add_argument('--lambda_recon', default=200, type=int, help='hyperparameters lambda_recon_loss')
parser.add_argument('--lambda_dim', default=5, type=int, help='hyperparameters lambda_dim_loss')
parser.add_argument('--lambda_stam', default=200, type=int, help='hyperparameters lambda_stam_loss')
parser.add_argument('--lambda_kl', default=0.02, type=int, help='hyperparameters lambda_stam_loss')


def siamese_setting(args):

    ckpt_name = args.description
    ckpt_dir = args.model_dir

    # backbone VAE of the instantGroup framework
    vae = VAE()
    vae.train().to(device0)
    opt_vae = Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # pretrained registration network
    reg = VxmDense.load('')
    reg.eval().to(device1)


    for epoch in range(args.num_epoch):

        training_generator = Data.DataLoader(Dataset_mri(path=args.data_dir), batch_size=args.bs, shuffle=True, num_workers=2)
        loss_list = [[], [], [], [], []]  # recon, kl, dim, stam, overall

        for iter, (input1, input2) in enumerate(tqdm(training_generator)):

            with autocast():

                input1 = input1.to(device0)
                input2 = input2.to(device0)

                recon_output1, latent_z1 = vae(input1)
                recon_output2, latent_z2 = vae(input2)

                # Construct the atlas from the average of the two latent variables
                z_mu1, z_var1 = torch.chunk(latent_z1, chunks=2, dim=1)
                z_mu2, z_var2 = torch.chunk(latent_z2, chunks=2, dim=1)
                atlas_latent = (vae.sample_z(z_mu1, z_var1) + vae.sample_z(z_mu2, z_var2)) * 0.5
                recon_atlas = vae.decode(atlas_latent.to(device0))

                # Perform registration from each input to the atlas
                warped_input1, warped_atlas1, flow1 = reg(input1.to(device1), recon_atlas.to(device1), True)
                warped_input2, warped_atlas2, flow2 = reg(input2.to(device1), recon_atlas.to(device1), True)

                # Loss Computation

                # Dual VAE - reconstruction loss and KL loss
                recon_loss1 = (recon_output1.to(device1) - input1.to(device1)) ** 2
                recon_loss2 = (recon_output2.to(device1) - input2.to(device1)) ** 2
                recon_loss = (torch.mean(recon_loss1) + torch.mean(recon_loss2))

                kl_loss1 = torch.mean(0.5 * torch.sum(torch.exp(z_var1) + z_mu1 ** 2 - 1. - z_var1, 1))
                kl_loss2 = torch.mean(0.5 * torch.sum(torch.exp(z_var2) + z_mu2 ** 2 - 1. - z_var2, 1))

                kl_loss = kl_loss1.to(device1) + kl_loss2.to(device1)

                loss_list[0].append(recon_loss.item())
                loss_list[1].append(kl_loss.item())

                # Displacement Inversion Loss
                dim_loss = torch.mean((flow1 + flow2) ** 2).to(device1)
                loss_list[2].append(dim_loss.item())

                # Subject-Template Alignment Loss
                stam_loss = torch.mean((warped_atlas1.to(device0) - input1) ** 2 + \
                              (warped_atlas2.to(device0)- input2) ** 2)
                loss_list[3].append(stam_loss.item())

                loss = recon_loss * args.lambda_recon + kl_loss * args.lambda_kl + dim_loss * args.lambda_dim + \
                       stam_loss.to(device1) * args.lambda_stam
                loss_list[4].append(loss.item())

                opt_vae.zero_grad()
                loss.backward()
                opt_vae.step()

                if iter % 30 == 0:
                    print(' '.join(['%.4e' % np.mean(loss_value) for loss_value in loss_list]))
                    loss_list = [[], [], [], [], []]  # recon, kl, dim, stam, overall

            # clear cache
            del z_mu1, z_mu2, z_var1, z_var2, recon_atlas, recon_output1, recon_output2, atlas_latent, loss, recon_loss, kl_loss, stam_loss, dim_loss
            torch.cuda.empty_cache()

            if iter % 500 == 0:
                save_checkpoint(vae, ckpt_dir, ckpt_name='_'.join([ckpt_name, str(epoch), str(iter)]))

        # final model save
        save_checkpoint(vae, ckpt_dir, ckpt_name=str(epoch))


def save_checkpoint(model, ckpt_dir, ckpt_name):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"model_{ckpt_name}.pt")
    torch.save(model.state_dict(), path)
    return


def train(args):
    # Initialization
    set_seed(0)

    # Setting GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    global device0, device1
    gpu0, gpu1 = '0', '1'
    device0 = torch.device('cuda:' + gpu0)
    device1 = torch.device('cuda:' + gpu1)

    siamese_setting(args)


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)


