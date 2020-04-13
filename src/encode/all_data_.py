import csv
import collections
import operator
from csv import DictReader
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import os
'''
 按照原始数据集划分
'''

def to_libsvm_encode(datapath):
    print('###### to libsvm encode ######\n')
    train_encode = datapath + 'train.txt'
    new_train_file = datapath + 'train_.txt'

    test_encode = datapath + 'test.txt'
    new_test_file = datapath + 'test_.txt'

    train_list = list()
    with open(new_train_file, 'w') as new_train_f_out:
        for line in open(train_encode):
            # print(line.strip().replace(':1', '').replace('\t', ','))
            new_train_f_out.write(line.strip().replace(':1', '').replace('\t', ',') + '\n')
            # train_list.append(line.strip().replace(':1', '').split(' '))

    # new_train_encode = pd.DataFrame(data=np.array(train_list).astype(int))
    # new_train_encode.to_csv(new_train_file, header=None, index=None)

    test_list = list()
    with open(new_test_file, 'w') as new_test_f_out:
        for line in open(test_encode):
            new_test_f_out.write(line.strip().replace(':1', '').replace('\t', ',') + '\n')
            # test_list.append(line.strip().replace(':1', '').split(' '))

    # new_test_encode = pd.
            # (new_test_file, header=None, index=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/')
    parser.add_argument('--dataset_name', default='avazu/', help='ipinyou, cretio, yoyi, avazu')
    parser.add_argument('--campaign_id', default='avazu/', help='1458, 3386, 3358, 3427, 3476, avazu')
    parser.add_argument('--is_to_csv', default=False)
    parser.add_argument('--is_separate_data', default=True)

    args = parser.parse_args()

    data_path = args.data_path + args.dataset_name + args.campaign_id

    to_libsvm_encode(data_path)

