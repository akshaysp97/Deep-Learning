import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
Config['root_path'] = '/content/polyvore_outfits'
#Config['root_path_aws'] = '/home/ubuntu/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = ''


Config['use_cuda'] = True
Config['debug'] = False
Config['num_epochs'] = 2
Config['batch_size'] = 128
Config['n_class'] = 2
Config['test_file'] = 'test_category_hw.txt'
Config['root_path_aws'] = '/content/polyvore_outfits'
Config['aws'] = True
Config['compatible_train'] = 'pairwise_compatibility_train.txt'
Config['compatible_valid'] = 'pairwise_compatibility_valid.txt'
Config['debug_size'] = 100

Config['learning_rate'] = 0.00005
Config['num_workers'] = 1