import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import models
import data.poison_cifar as poison
from architectures import DENOISERS_ARCHITECTURES, get_architecture, IMAGENET_CLASSIFIERS, AUTOENCODER_ARCHITECTURES

parser = argparse.ArgumentParser(description='Train poisoned networks')

# Basic model parameters.
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
parser.add_argument('--checkpoint', type=str, default='./save/model_39.th', help='The checkpoint to be pruned')
parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
parser.add_argument('--print-every', type=int, default=500, help='print results every few iterations')
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
parser.add_argument('--output-dir', type=str, default='./save')

parser.add_argument('--trigger-info', type=str, default='./save/trigger_info.th', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--anp-eps', type=float, default=0.4)
parser.add_argument('--anp-steps', type=int, default=1)
parser.add_argument('--anp-alpha', type=float, default=0.2)

# Dataset
parser.add_argument('--dataset', type=str, default='cifar10')

# Optimization Method
parser.add_argument('--optimization_method', default='FO', type=str,
                    help="FO: First-Order (White-Box), ZO: Zeroth-Order (Black-box)",
                    choices=['FO', 'ZO'])
parser.add_argument('--zo_method', default='RGE', type=str,
                    help="Random Gradient Estimation: RGE, Coordinate-Wise Gradient Estimation: CGE",
                    choices=['RGE', 'CGE', 'CGE_sim'])
parser.add_argument('--q', default=192, type=int, metavar='N',
                    help='query direction (default: 20)')
parser.add_argument('--mu', default=0.005, type=float, metavar='N',
                    help='Smoothing Parameter')

# Model type
parser.add_argument('--model_type', default='AE_DS', type=str,
                    help="Denoiser + (AutoEncoder) + classifier/reconstructor",
                    choices=['DS', 'AE_DS'])
parser.add_argument('--encoder_arch', type=str, default='cifar_encoder_192_24_noise', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--decoder_arch', type=str, default='cifar_decoder_192_24_noise', choices=AUTOENCODER_ARCHITECTURES)
parser.add_argument('--classifier', default='', type=str,
                    help='path to the classifier used with the `classificaiton`'
                         'or `stability` objectives of the denoiser.')
# parser.add_argument('--pretrained_denoiser', default='./trained_models/CIFAR-10/ZO_AE_DS_lr-3_q192_Coord/best_denoiser.pth.tar', type=str, help='path to a pretrained denoiser')
parser.add_argument('--pretrained_denoiser', default='', type=str, help='path to a pretrained denoiser')
parser.add_argument('--pretrained_encoder', default='./trained_models/CIFAR-10/ZO_AE_DS_lr-3_q192_Coord/best_encoder.pth.tar', type=str, help='path to a pretrained encoder')
parser.add_argument('--pretrained_decoder', default='./trained_models/CIFAR-10/ZO_AE_DS_lr-3_q192_Coord/best_decoder.pth.tar', type=str, help='path to a pretrained decoder')


args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
    if args.trigger_info:
        trigger_info = torch.load(args.trigger_info, map_location=device)
    else:
        if args.poison_type == 'benign':
            trigger_info = None
        else:
            triggers = {'badnets': 'checkerboard_1corner',
                        'clean-label': 'checkerboard_4corner',
                        'blend': 'gaussian_noise'}
            trigger_type = triggers[args.poison_type]
            pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
            trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
                            'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}

    orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    _, clean_val = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
                                        perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    # _, clean_val = poison.split_dataset(dataset=orig_train, val_frac=0.1,
                                        # perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)

    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                                   num_samples=args.print_every * args.batch_size)
    clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
                                  shuffle=False, sampler=random_sampler, num_workers=0)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device)
    # net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d)
    net = getattr(models, args.arch)(num_classes=10)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # --------------------- Model Loading -------------------------
    # a) Denoiser
    if args.pretrained_denoiser:
        checkpoint = torch.load(args.pretrained_denoiser)
        denoiser = get_architecture(checkpoint['arch'], args.dataset)
        denoiser.load_state_dict(checkpoint['state_dict'])
    else:
        denoiser = get_architecture('cifar_dncnn_perturb', args.dataset)

    # b) AutoEncoder
    if args.model_type == 'AE_DS':
        if args.pretrained_encoder:
            checkpoint = torch.load(args.pretrained_encoder, map_location=device)
            encoder = get_architecture(checkpoint['arch'], args.dataset)
            encoder.load_state_dict(checkpoint['state_dict'])
        else:
            encoder = get_architecture(args.encoder_arch, args.dataset)

        if args.pretrained_decoder:
            checkpoint = torch.load(args.pretrained_decoder, map_location=device)
            decoder = get_architecture(checkpoint['arch'], args.dataset)
            decoder.load_state_dict(checkpoint['state_dict'])
        else:
            decoder = get_architecture(args.decoder_arch, args.dataset)
        import torch.nn as nn
        def modify_model(model):
            new_model = nn.Sequential()
            for name, layer in model.named_children():
                if isinstance(layer, nn.Conv2d):
                    # Add the convolutional layer
                    new_model.add_module(name, layer)
                    # Add a new batch normalization layer
                    new_model.add_module(f'{name}_bn', models.NoisyBatchNorm2d(layer.out_channels))
                else:
                    new_model.add_module(name, layer)
            return new_model
        # encoder.encoder = modify_model(encoder.encoder)
        # decoder.decoder = modify_model(decoder.decoder)

        encoder = encoder.to(device)
        decoder = decoder.to(device)

    parameters = list(denoiser.named_parameters()) 
    mask_params = [v for n, v in parameters if "neuron_mask" in n]
    mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
    noise_params = [v for n, v in parameters if "neuron_noise" in n]
    noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

    # Step 3: train backdoored models
    print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
    # nb_repeat = 500
    for i in range(1):
        start = time.time()
        lr = mask_optimizer.param_groups[0]['lr']
        train_loss, train_acc = mask_train(models=[denoiser, encoder, decoder, net], criterion=criterion, data_loader=clean_val_loader,
                                           mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        cl_test_loss, cl_test_acc = test(models=[denoiser, encoder, decoder, net], criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(models=[denoiser, encoder, decoder, net], criterion=criterion, data_loader=poison_test_loader)
        end = time.time()
        print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc))
    state_dict = encoder.state_dict()
    state_dict.update(decoder.state_dict())
    save_mask_scores(state_dict, os.path.join(args.output_dir, 'mask_values.txt'))


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=args.anp_eps)


def mask_train(models, criterion, mask_opt, noise_opt, data_loader):
    denoiser, encoder, decoder, model = models
    denoiser.train()
    # encoder.train()
    # decoder.train()
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if args.anp_eps > 0.0:
            reset(denoiser, rand_init=True)
            for _ in range(args.anp_steps):
                noise_opt.zero_grad()

                include_noise(denoiser)
                output_noise = denoiser(images)
                output_noise = model(output_noise)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(denoiser)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if args.anp_eps > 0.0:
            include_noise(denoiser)
            output_noise = denoiser(images)
            output_noise = model(output_noise)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(denoiser)
        output_clean = denoiser(images)
        output_clean = model(output_clean)
        loss_nat = criterion(output_clean, labels)
        loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(denoiser)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(models, criterion, data_loader):
    denoiser, encoder, decoder, model = models
    denoiser.eval()
    encoder.eval()
    decoder.eval()
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(denoiser(images))
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)


if __name__ == '__main__':
    main()
