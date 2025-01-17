"""
    Plot the optimization path in the space spanned by principle directions.
"""

import numpy as np
import torch
import copy
import math
import h5py
import os
import argparse
import model_loader
import net_plotter
from projection import setup_PCA_directions_compare, project_trajectory_compare
import plot_2D


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot optimization trajectory')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--model', default='resnet56', help='trained models')
    parser.add_argument('--model_folder1', default='', help='folders for models to be projected')
    parser.add_argument('--model_folder2', default='', help='folders for models to be projected')
    parser.add_argument('--dir_type', default='weights',
        help="""direction type: weights (all weights except bias and BN paras) |
                                states (include BN.running_mean/var)""")
    parser.add_argument('--ignore', default='', help='ignore bias and BN paras: biasbn (no bias or bn)')
    parser.add_argument('--prefix', default='model_', help='prefix for the checkpint model')
    parser.add_argument('--suffix', default='.t7', help='prefix for the checkpint model')
    parser.add_argument('--start_epoch', default=0, type=int, help='min index of epochs')
    parser.add_argument('--max_epoch', default=300, type=int, help='max number of epochs')
    parser.add_argument('--save_epoch', default=1, type=int, help='save models every few epochs')
    parser.add_argument('--dir_file', default='', help='load the direction file for projection')
    parser.add_argument('--loss_file1', default='', help='load the direction file for projection')
    parser.add_argument('--loss_file2', default='', help='load the direction file for projection')
    parser.add_argument('--annotate_every_n1', default=5, help='load the direction file for projection')
    parser.add_argument('--annotate_every_n2', default=5, help='load the direction file for projection')


    args = parser.parse_args()
    with open(args.loss_file1, 'r') as f:
        losses1 = [float(line.strip()) for line in f.readlines()]

    losses1 = [f'{loss:.2f}' for loss in losses1]

    with open(args.loss_file2, 'r') as f:
        losses2 = [float(line.strip()) for line in f.readlines()]

    losses2 = [f'{loss:.2f}' for loss in losses2]

    #--------------------------------------------------------------------------
    # load the final model
    #--------------------------------------------------------------------------
    last_model_file1 = args.model_folder1 + '/' + args.prefix + str(args.max_epoch) + args.suffix
    last_model_file2 = args.model_folder2 + '/' + args.prefix + str(args.max_epoch) + args.suffix
    net1 = model_loader.load(args.dataset, args.model, last_model_file1)
    net2 = model_loader.load(args.dataset, args.model, last_model_file2)
    w1 = net_plotter.get_weights(net1)
    w2 = net_plotter.get_weights(net2)
    s1 = net1.state_dict()
    s2 = net2.state_dict()

    #--------------------------------------------------------------------------
    # collect models to be projected
    #--------------------------------------------------------------------------
    model_files1 = []
    for epoch in range(args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch):
        model_file = args.model_folder1 + '/' + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files1.append(model_file)
    model_files2 = []
    for epoch in range(args.start_epoch, args.max_epoch + args.save_epoch, args.save_epoch):
        model_file = args.model_folder2 + '/' + args.prefix + str(epoch) + args.suffix
        assert os.path.exists(model_file), 'model %s does not exist' % model_file
        model_files2.append(model_file)

    #--------------------------------------------------------------------------
    # load or create projection directions
    #--------------------------------------------------------------------------
    if args.dir_file:
        dir_file = args.dir_file
    else:
        dir_file = setup_PCA_directions_compare(args, model_files1, model_files2, w1, w2, s1, s2, args.model_folder2)
    #--------------------------------------------------------------------------
    # projection trajectory to given directions
    #--------------------------------------------------------------------------
    proj_file1 = project_trajectory_compare(dir_file, w1, s1, args.dataset, args.model,
                                model_files1, args.dir_type, 'cos', 1)
    proj_file2 = project_trajectory_compare(dir_file, w2, s2, args.dataset, args.model,
                                model_files2, args.dir_type, 'cos', 2)
    plot_2D.plot_trajectory_compare(proj_file1, proj_file2, dir_file, args.model_folder1.split("/")[-1], args.model_folder2.split("/")[-1], losses1, losses2, int(args.annotate_every_n1), int(args.annotate_every_n2))