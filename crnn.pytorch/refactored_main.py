from __future__ import print_function

import argparse
import datetime
import os, sys
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from warpctc_pytorch import CTCLoss

import dataset
import utils
from model_code import train_batch, val_batch, run_net_batch
from models import crnn as crnn_model

import json


def load_trained_crnn_for_eval(opt):
    opt.crnn = 'trained_models/netCRNN_43_500_0.667363636364.pth'
    opt.batchSize = 20

    cudnn.benchmark = True

    nclass = len(opt.alphabet) + 1
    nc = 1

    crnn = crnn_model.CRNN(opt.imgH, nc, nclass, opt.nh)

    if opt.crnn == '':
        raise Exception("Can't validate if you won't give me a model")

    # print('Loading pre-trained model from %s' % opt.crnn)
    loaded_model = torch.load(opt.crnn, map_location=lambda storage, loc: storage)

    # print("Loaded model accuracy: " + str(loaded_model['accuracy']))
    # print("Loaded model epoch: " + str(loaded_model['epoch']))
    loaded_state_dict = loaded_model['state']

    new_net_state = crnn.state_dict()

    for param_name in loaded_state_dict:
        new_name = param_name[7:]
        if new_name in new_net_state:
            new_net_state[new_name] = loaded_state_dict[param_name]

    crnn.load_state_dict(new_net_state)

    for p in crnn.parameters():
        p.requires_grad = False

    crnn.eval()

    return crnn, utils.strLabelConverter(opt.alphabet), CTCLoss()


def main(opt, case):
    print("Arguments are : " + str(opt))

    if opt.experiment is None:
        opt.experiment = 'expr'
    os.system('mkdir {0}'.format(opt.experiment))

    # Why do we use this?
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

        opt.cuda = True
        print('Set CUDA to true.')

    train_dataset = dataset.hwrDataset(mode="train")
    assert train_dataset

    # The shuffle needs to be false when the sizing has been done.

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=True))

    test_dataset = dataset.hwrDataset(mode="test", transform=dataset.resizeNormalize((100, 32)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    criterion = CTCLoss()

    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    crnn = crnn_model.CRNN(opt.imgH, nc, nclass, opt.nh)
    crnn.apply(weights_init)

    if opt.cuda and not opt.uses_old_saving:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        criterion = criterion.cuda()

    if opt.crnn != '':

        print('Loading pre-trained model from %s' % opt.crnn)
        loaded_model = torch.load(opt.crnn)

        if opt.uses_old_saving:
            print("Assuming model was saved in rudementary fashion")
            crnn.load_state_dict(loaded_model)
            crnn.cuda()

            crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
            criterion = criterion.cuda()
            start_epoch = 0
        else:
            print("Loaded model accuracy: " + str(loaded_model['accuracy']))
            print("Loaded model epoch: " + str(loaded_model['epoch']))
            start_epoch = loaded_model['epoch']
            crnn.load_state_dict(loaded_model['state'])

    # Read this.
    loss_avg = utils.averager()

    # If following the paper's recommendation, using AdaDelta
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    elif opt.adagrad:
        print("Using adagrad")
        optimizer = optim.Adagrad(crnn.parameters(), lr=opt.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    converter = utils.strLabelConverter(opt.alphabet)

    best_val_accuracy = 0

    for epoch in range(start_epoch, opt.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = train_batch(crnn, criterion, optimizer, train_iter, opt, converter)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.niter, i, len(train_loader), loss_avg.val()) + " " + case)
                loss_avg.reset()

            if i % opt.valInterval == 0:
                try:
                    val_loss_avg, accuracy = val_batch(crnn, opt, test_dataset, converter, criterion)

                    model_state = {
                        'epoch': epoch + 1,
                        'iter': i,
                        'state': crnn.state_dict(),
                        'accuracy': accuracy,
                        'val_loss_avg': val_loss_avg,
                    }
                    utils.save_checkpoint(model_state, accuracy > best_val_accuracy,
                                          '{0}/netCRNN_{1}_{2}_{3}.pth'.format(opt.experiment, epoch, i, accuracy),
                                          opt.experiment)

                    if accuracy > best_val_accuracy:
                        best_val_accuracy = accuracy

                except Exception as e:
                    print(e)


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', help='path to dataset')
    parser.add_argument('--valroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input vto network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--adagrad', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--uses_old_saving', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()
    opt.adadelta = True

    return opt


def extract_result(opt, crnn, converter, extra_path):
    print(extra_path)

    test_dataset = dataset.hwrDataset(mode="test", transform=dataset.resizeNormalize((100, 32)),
                                      return_index=True, extra_path=extra_path)

    accuracy, predicted_list = run_net_batch(crnn, opt, test_dataset, converter)

    sorted(predicted_list, key=lambda instance: instance.index)

    return predicted_list


def validate(opt, crnn, converter, criterion):
    test_dataset = dataset.hwrDataset(mode="test", transform=dataset.resizeNormalize((100, 32)))

    val_loss_avg, accuracy, corrected_accuracy = val_batch(crnn, opt, test_dataset, converter, criterion, full_val=True)

    print(corrected_accuracy)
    print(accuracy)
    print(val_loss_avg)


# Note - Both Adam and RMS Prop have not performed well
if __name__ == '__main__':
    opt = get_parameters()

    case = "double_batch_scene_pretrained"
    case = "double_batch_adaGrad_from_adaDelta_which_pretrained_crnn"
    case = "validate"

    if case == 'double_batch_adaGrad_from_adaDelta_which_pretrained_crnn':
        opt.crnn = 'iam/copied_from_laptop_2017_11_15_01_35/model_best.pth.tar'
        opt.adagrad = True
        opt.adadelta = False
        opt.batchSize = 110

    elif case == "double_batch_scene_pretrained":
        opt.crnn = 'trained_models/pretrained_crnn.pth'
        opt.uses_old_saving = True
        opt.batchSize = 110

    elif case == "validate":
        crnn, converter, criterion = load_trained_crnn_for_eval(opt)
        validate(opt, crnn, converter, criterion)
        sys.exit(1)

    experiment_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    opt.experiment = os.path.join("iam", case + "_" + experiment_start_time)

    os.system('mkdir {0}'.format(opt.experiment))
    json.dump(vars(opt), open(os.path.join(opt.experiment, experiment_start_time + "arguments.json"), "w"))

    main(opt, case)