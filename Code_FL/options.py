#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments   
    parser.add_argument('--epochs', type=int, default=30, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: N")#10
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--num_items_train', type=int, default=3000, help="dataset size for each user")
    parser.add_argument('--num_items_test', type=int, default=10000, help="dataset size for each user")
    parser.add_argument('--val', type=bool, default=False, help="whether using validation dataset")      
    parser.add_argument('--ratio_val', type=float, default=0.2, help="ratio of validation dataset")      
    parser.add_argument('--local_bs', type=int, default=110, help="local batch size: B")#110
    parser.add_argument('--lr_orig', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_g_orig', type=float, default=0.05, help='global learning rate')#0.05
    parser.add_argument('--beta_1', type=float, default=0.9, help='momentum parameter')
    parser.add_argument('--beta_2', type=float, default=0.99, help='momentum parameter')    
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.9)')
    parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d. or not, if set non-i.i.d., please reduce the number of user samples')
    parser.add_argument('--degree_noniid', type=float, default=1.0, help='the degree of non-i.i.d.')
    parser.add_argument('--ratio_train', type=list, default=[1], help="distribution of training datasets")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")#cifar
    parser.add_argument('--data_dir', type=str, default='D:/WORK/Scholar/Datasets', help="dir of datasets")
    parser.add_argument('--batch_type', type=str, default='BSGD', help="BSGD or mini-BSGD")

    parser.add_argument('--acceleration', type=bool, default=False)
 
    # other arguments
    parser.add_argument('--lr_decay', default=True, help="Learning rate decay")
    parser.add_argument('--lr_decay_rate', default=0.985)
    parser.add_argument('--num_experiments', type=int, default=1, help="number of experiments")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=-1)

    # key argements
    parser.add_argument('--read_text_name', default='number_data3')#number_data3
    parser.add_argument('--num_algo', type=int, default=2)
    parser.add_argument('--set_algo_type', type=list, default=[0,1], help="The selected algorithm") 
    parser.add_argument('--prox', type=bool, default=False, help="whether using prox")
    parser.add_argument('--mu', type=float, default=0.1, help="whether using prox")        
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        args.set_epochs = [50]
        args.set_num_users = [5]
        args.lr_orig = 0.05
        args.lr_g_orig = 0.05
        args.lr_decay_rate = 0.9
    elif args.dataset == 'FashionMNIST':
        args.set_epochs = [200]
        args.set_num_users = [5]
        args.lr_orig = 0.05
        args.lr_g_orig = 0.05
        args.lr_decay_rate = 0.9
    elif args.dataset == 'cifar':
        args.set_epochs = [2500]
        args.set_num_users = [10]
        args.lr_orig = 0.08
        args.lr_g_orig = 0.08
        args.lr_decay_rate = 0.985
    elif args.dataset == 'femnist':
        args.set_epochs = [50]
        args.set_num_users = [5]
        args.lr_orig = 0.05
        args.lr_g_orig = 0.05
        args.lr_decay_rate = 0.9 

    return args