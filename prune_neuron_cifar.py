import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
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
parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
parser.add_argument('--output-dir', type=str, default='./save')

parser.add_argument('--trigger-info', type=str, default='./save/trigger_info.th', help='The information of backdoor trigger')
parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
                    help='type of backdoor attacks for evaluation')
parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

parser.add_argument('--mask-file', type=str, default='./save/mask_values.txt', help='The text file containing the mask values')
parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
parser.add_argument('--pruning-max', type=float, default=0.90, help='the maximum number/threshold for pruning')
parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')

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
parser.add_argument('--pretrained-denoiser', default='', type=str, help='path to a pretrained denoiser')
parser.add_argument('--pretrained-encoder', default='./trained_models/CIFAR-10/AutoEncoder_192_24_StanTrain_lr1e-3/encoder.pth.tar', type=str, help='path to a pretrained encoder')
parser.add_argument('--pretrained-decoder', default='./trained_models/CIFAR-10/AutoEncoder_192_24_StanTrain_lr1e-3/decoder.pth.tar', type=str, help='path to a pretrained decoder')

args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # Step 1: create poisoned / clean test set
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

    clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

    # Step 2: load model checkpoints and trigger info
    state_dict = torch.load(args.checkpoint, map_location=device)
    # net = getattr(models, args.arch)(num_classes=10, norm_layer=models.NoisyBatchNorm2d)
    net = getattr(models, args.arch)(num_classes=10)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

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
        encoder.encoder = modify_model(encoder.encoder)
        decoder.decoder = modify_model(decoder.decoder)

        encoder = encoder.to(device)
        decoder = decoder.to(device)

    # Step 3: pruning
    mask_values = read_data(args.mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(models=[encoder, decoder, net], criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(models=[encoder, decoder, net], criterion=criterion, data_loader=poison_test_loader)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
    if args.pruning_by == 'threshold':
        results = evaluate_by_threshold(
            net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
        )
    else:
        results = evaluate_by_number(
            net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
            criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
        )
    file_name = os.path.join(args.output_dir, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results


def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results


def test(models, criterion, data_loader):
    encoder, decoder, net = models
    encoder.eval()
    decoder.eval()
    net.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = encoder(images)
            output = decoder(images)
            output = net(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()

