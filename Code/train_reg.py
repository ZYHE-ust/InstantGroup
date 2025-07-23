import argparse
from tqdm import tqdm
from utils import *
from reg_model import VxmDense
import os


# parse the commandline
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='./checkpoints/',
                    help='model output directory (default: models)')
parser.add_argument('--dimension', type=int, default=3, help='dimension of the scan')

# training parameters
parser.add_argument('--image_loss', default='ncc')
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--bs', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--datapath', default='/home/albert/zy/vgr/data/OASIS/', type=str, help='data path for training images')
parser.add_argument('--load-model', default='', help='optional model file to initialize with')
parser.add_argument('--initial_epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', type=bool, default=True, help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.1,
                    help='weight of deformation loss (default: 0.01)')

def train(args):

    ckpt_name = 'reg_model'
    inshape = [96, 112, 96]
    bidir = args.bidir

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

    # model initialization
    model = VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize)

    # model loading
    # model = VxmDense.load('', device)

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = ncc_loss
    elif args.image_loss == 'mse':
        image_loss_func = mse_loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # two image loss functions if bidirectional
    loss_func = [image_loss_func, image_loss_func] if bidir else [image_loss_func]
    weights = [0.5, 0.5] if bidir else [1]

    # prepare deformation loss
    loss_func += [gradient_loss]
    weights += [args.weight]

    # training epochs
    for epoch in range(args.epochs):

        epoch_loss = []
        epoch_total_loss = []

        training_generator = Data.DataLoader(Dataset_mri(path=args.datapath), batch_size=args.bs, shuffle=True,
                                             num_workers=2)

        for step in tqdm(range(args.steps_per_epoch)):
            # iterate over the data generator
            inputs, targets = next(iter(training_generator))
            inputs, targets = inputs.to(device), targets.to(device)

            # run inputs through the model to produce a warped image and flow field
            targets_pred = model(inputs, targets, inshape=inshape)

            # calculate total loss
            loss_list = []
            curr_loss = loss_func[0](targets, targets_pred[0], None, device=device) * weights[0]
            loss_list.append(curr_loss.item())
            loss = curr_loss

            if bidir:
                # bidirectional registration
                curr_loss = loss_func[1](inputs, targets_pred[1], None, device=device) * weights[1]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            curr_loss = loss_func[-1](targets_pred[-1]) * weights[-1]
            loss_list.append(curr_loss.item())
            loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model.save(os.path.join(model_dir, ckpt_name + '_epoch_%04d.pt' % epoch))
            print(' '.join(('Epoch %d/%d ' % (epoch + 1, args.epochs),
                            ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)]))), flush=True)

    # final model save
    model.save(os.path.join(model_dir, ckpt_name + '_%04d.pt' % args.epochs))


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
