"""!
@brief Infer Dataset Specific parameters and return generators

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
from __config__ import WHAM_ROOT_PATH, LIBRI2MIX_ROOT_PATH, ESC50_ROOT_PATH
import self_supervised_separation.dataloaders.libri2mix as libri2mix_loader
import self_supervised_separation.dataloaders.wham as wham_loader
import self_supervised_separation.dataloaders.augmented_mix_dataloader as augmented_mix_dataloader


def create_loader_for_simple_dataset(dataset_name=None,
                                     separation_task=None,
                                     data_split=None,
                                     sample_rate=None,
                                     min_or_max=None,
                                     zero_pad=None,
                                     timelegth=None,
                                     normalize_audio=None,
                                     n_samples=None,
                                     batch_size=None,
                                     num_workers=None,
                                     max_abs_snr=None):
    translated_split = None

    if dataset_name == 'WHAM':
        loader = wham_loader
        root_path = WHAM_ROOT_PATH
        translator = {'train': 'tr', 'test': 'tt', 'val': 'cv'}
        translated_split = translator[data_split]
    elif dataset_name == 'ESC50':
        if data_split == 'train':
            fixed_seed = 0
        elif data_split == 'test':
            fixed_seed = 8
        else:
            fixed_seed = 9
        data_loader = augmented_mix_dataloader.AugmentedOnlineMixingDataset(
            input_dataset_p=[os.path.join(ESC50_ROOT_PATH, data_split)],
            datasets_priors=[1.],
            batch_size=batch_size,
            n_jobs=num_workers,
            n_samples=n_samples,
            return_items=['wav'],
            fs=float(sample_rate),
            selected_timelength=timelegth,
            n_sources=2,
            normalize_audio=normalize_audio,
            max_abs_snr=max_abs_snr,
            fixed_seed=fixed_seed)
        return data_loader
    elif dataset_name == 'LIBRI2MIX':
        loader = libri2mix_loader
        root_path = LIBRI2MIX_ROOT_PATH
        if n_samples > 13900 and data_split == 'train':
            print('Going to use train-360 for training LibriMix...')
            translated_split = 'train-360'
        elif n_samples <= 13900 and data_split == 'train':
            print('Going to use train-100 for training LibriMix...')
            translated_split = 'train-100'
        elif data_split == 'test':
            translated_split = 'test'
        elif data_split == 'val':
            translated_split = 'dev'
    else:
        raise ValueError('Dataset: {} is not yet supported!'.format(
            dataset_name))

    data_loader = loader.Dataset(
        root_dirpath=root_path, task=separation_task,
        split=translated_split, sample_rate=sample_rate, timelength=timelegth,
        zero_pad=zero_pad, min_or_max=min_or_max,
        augment='tr' in data_split,
        normalize_audio=normalize_audio, n_samples=n_samples)
    return data_loader

def setup(hparams):
    # Create all generators
    generators = {}
    for data_split in ['train', 'val', 'test', 'train_val']:
        if hparams[data_split] is None:
            generators[data_split] = None
            continue

        if len(hparams[data_split]) > 1:
            raise ValueError('Current implementation does not support '
                             'training using multiple datasets.')

        loader = create_loader_for_simple_dataset(
                    dataset_name=hparams[data_split][0],
                    separation_task=hparams['separation_task'],
                    data_split=data_split.split('_')[0],
                    sample_rate=hparams['fs'],
                    min_or_max=hparams['min_or_max'],
                    zero_pad=hparams['zero_pad_audio'],
                    timelegth=hparams['audio_timelength'],
                    normalize_audio=hparams['normalize_audio'],
                    n_samples=hparams['n_'+data_split],
                    batch_size=hparams['batch_size'],
                    num_workers=hparams['n_jobs'],
                    max_abs_snr=hparams['max_abs_snr'])
        generators[data_split] = loader.get_generator(
            batch_size=hparams['batch_size'], num_workers=hparams['n_jobs'])

    return generators
