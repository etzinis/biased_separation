"""!
@brief Running supervised SudORM-RF with variance reduction

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

from __config__ import API_KEY
from comet_ml import Experiment

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from pprint import pprint
import self_supervised_separation.dnn.experiments.utils.cmd_args_parser as parser
import self_supervised_separation.dnn.experiments.utils.dataset_setup as dataset_setup
import self_supervised_separation.dnn.losses.sisdr as sisdr_lib
import self_supervised_separation.dnn.models.sudormrf as sudormrf
import self_supervised_separation.dnn.utils.cometml_loss_report as cometml_report
import self_supervised_separation.dnn.utils.metrics_logger as \
    cometml_assets_logger
import self_supervised_separation.dnn.utils.cometml_log_audio as cometml_audio_logger


args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.setup(hparams)

if hparams['separation_task'] == 'enh_single':
    hparams['n_sources'] = 1
else:
    hparams['n_sources'] = 2

# if hparams["log_audio"]:
audio_logger = cometml_audio_logger.AudioLogger(
    fs=hparams["fs"], bs=hparams["batch_size"], n_sources=hparams["n_sources"])


experiment = Experiment(API_KEY, project_name=hparams["project_name"])
experiment.log_parameters(hparams)
experiment_name = '_'.join(hparams['cometml_tags'])
for tag in hparams['cometml_tags']:
    experiment.add_tag(tag)
if hparams['experiment_name'] is not None:
    experiment.set_name(hparams['experiment_name'])
else:
    experiment.set_name(experiment_name)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
    [cad for cad in hparams['cuda_available_devices']])

back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_SISDRi',
    # sisdr_lib.PermInvariantSISDR(batch_size=hparams['batch_size'],
    #                              n_sources=hparams['n_sources'],
    #                              zero_mean=True,
    #                              backward_loss=True,
    #                              improvement=True)
    sisdr_lib.HigherOrderPermInvariantSISDR(batch_size=hparams['batch_size'],
                                            n_sources=hparams['n_sources'],
                                            zero_mean=True,
                                            backward_loss=True,
                                            improvement=True,
                                            var_weight=0.0)
)

val_losses = {}
all_losses = []
for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue
    val_losses[val_set] = {}
    all_losses.append(val_set + '_SISDR')
    val_losses[val_set][val_set + '_SISDR'] = sisdr_lib.PermInvariantSISDR(
        batch_size=hparams['batch_size'], n_sources=hparams['n_sources'],
        zero_mean=True, backward_loss=False, improvement=True,
        return_individual_results=True)
all_losses.append(back_loss_tr_loss_name)

histogram_names = ['tr_input_snr']
eval_generators_names = [x for x in generators
                         if not x == 'train' and generators[x] is not None]
for val_set in eval_generators_names:
    if generators[val_set] is None:
        continue
    histogram_names += [
        val_set+'_input_snr', val_set+'_SISDRi', val_set+'_SISDR']

model = sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                          in_channels=hparams['in_channels'],
                          num_blocks=hparams['num_blocks'],
                          upsampling_depth=hparams['upsampling_depth'],
                          enc_kernel_size=hparams['enc_kernel_size'],
                          enc_num_basis=hparams['enc_num_basis'],
                          num_sources=hparams['n_sources'])

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

model = torch.nn.DataParallel(model).cuda()
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def mix_with_random_snr(sources, abs_max_snr):
    new_sources = sources
    snr_ratio = (0.5 - torch.rand(sources.shape[0], 1)) * 2. * abs_max_snr
    snr_ratio = snr_ratio.cuda()
    new_energy_ratio = torch.sqrt(torch.pow(10., snr_ratio / 10.))
    energies = torch.sqrt(torch.sum(sources ** 2, dim=-1, keepdim=True))

    new_sources[:, 0] = new_energy_ratio * sources[:, 0] / (
            energies[:, 0] + 10e-8)
    new_sources[:, 1] = sources[:, 1] / (energies[:, 1] + 10e-8)
    return new_sources


tr_step = 0
val_step = 0
for i in range(hparams['n_epochs']):
    res_dic = {}
    histograms_dic = {}
    for loss_name in all_losses:
        res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        res_dic[loss_name+'i'] = {'mean': 0., 'std': 0., 'acc': []}
    for hist_name in histogram_names:
        histograms_dic[hist_name] = []
        histograms_dic[hist_name+'i'] = []
    print("Higher Order Sudo-RM-RF: {} - {} || Epoch: {}/{}".format(
        experiment.get_key(), experiment.get_tags(), i+1, hparams['n_epochs']))
    model.train()

    for data in tqdm(generators['train'], desc='Training'):
        opt.zero_grad()
        #m1wavs = data[0].cuda()
        clean_wavs = data[-1].cuda()

        if hparams['max_abs_snr'] > 0.:
            clean_wavs = mix_with_random_snr(clean_wavs, hparams['max_abs_snr'])

        histograms_dic['tr_input_snr'] += (10. * torch.log10(
            (clean_wavs[:, 0] ** 2).sum(-1) / (1e-8 + (
                    clean_wavs[:, 1] ** 2).sum(-1)))).tolist()

        # # Online mixing over samples of the batch. (This might cause to get
        # # utterances from the same speaker but it's highly improbable).
        energies = torch.sum(clean_wavs**2, dim=-1, keepdim=True)
        new_s1 = clean_wavs[:, 0, :]
        new_s2 = clean_wavs[torch.randperm(hparams['batch_size']), 1, :]
        new_s2 = new_s2 * torch.sqrt(energies[:, 1] /
                                      (new_s2**2).sum(-1, keepdims=True))

        m1wavs = normalize_tensor_wav(new_s1 + new_s2)
        clean_wavs[:, 0, :] = normalize_tensor_wav(new_s1)
        clean_wavs[:, 1, :] = normalize_tensor_wav(new_s2)
        # ===============================================

        # m1wavs = torch.sum(clean_wavs, dim=1)
        rec_sources_wavs = model(m1wavs.unsqueeze(1))

        l = back_loss_tr_loss(rec_sources_wavs,
                              clean_wavs,
                              epoch_count=i,
                              initial_mixtures=m1wavs.unsqueeze(1))
        if hparams['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           hparams['clip_grad_norm'])
        l.backward()
        opt.step()

    if hparams['reduce_lr_every'] > 0:
        if tr_step % hparams['reduce_lr_every'] == 0:
            new_lr = (hparams['learning_rate'] / (hparams['divide_lr_by'] ** (
                            tr_step // hparams['reduce_lr_every'])))
            print('Reducing Learning rate to: {}'.format(new_lr))
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
    tr_step += 1

    for val_set in [x for x in generators if not x == 'train']:
        if generators[val_set] is not None:
            model.eval()
            with torch.no_grad():
                for data in tqdm(generators[val_set], desc='Validation'):
                    # m1wavs = data[0].cuda()
                    clean_wavs = data[-1].cuda()

                    if hparams['max_abs_snr'] > 0.:
                        clean_wavs = mix_with_random_snr(clean_wavs,
                                                         hparams['max_abs_snr'])
                    m1wavs = torch.sum(clean_wavs, dim=1)
                    m1wavs = normalize_tensor_wav(m1wavs)

                    input_snr_tensor = 10. * torch.log10(
                        (clean_wavs[:, 0] ** 2).sum(-1) / (1e-8 + (
                                clean_wavs[:, 1] ** 2).sum(-1)))
                    input_snr_first = input_snr_tensor.tolist()
                    input_snr_second = (-input_snr_tensor).tolist()
                    histograms_dic[val_set + '_input_snr'] += [
                        val
                        for pair in zip(input_snr_first, input_snr_second)
                        for val in pair]


                    rec_sources_wavs = model(m1wavs.unsqueeze(1))

                    for loss_name, loss_func in val_losses[val_set].items():
                        l, l_improvement = loss_func(rec_sources_wavs,
                                      clean_wavs,
                                      initial_mixtures=m1wavs.unsqueeze(1))
                        values_in_list = l.tolist()
                        improvements_in_list = l_improvement.tolist()
                        res_dic[loss_name]['acc'] += values_in_list
                        res_dic[loss_name+'i']['acc'] += improvements_in_list
                        histograms_dic[loss_name] += values_in_list
                        histograms_dic[loss_name+'i'] += improvements_in_list
            audio_logger.log_batch(rec_sources_wavs, clean_wavs, m1wavs,
                                   experiment, step=val_step, tag=val_set)

    val_step += 1

    res_dic = cometml_report.report_losses_mean_and_std(
        res_dic, experiment, tr_step, val_step)
    cometml_report.report_histograms(
        histograms_dic, experiment, tr_step, val_step)
    scatter_lists = []
    for val_set in eval_generators_names:
        for suffix in ['', 'i']:
            scatter_lists.append([(val_set + '_input_snr',
                             histograms_dic[val_set + '_input_snr']),
                            (val_set + '_SISDR' + suffix,
                             histograms_dic[val_set + '_SISDR' + suffix])])
    cometml_report.report_scatterplots(
        scatter_lists, experiment, tr_step, val_step)
    # Save all metrics as assets.
    cometml_assets_logger.log_metrics(histograms_dic, '/tmp/', experiment,
                                      tr_step, val_step)
    cometml_assets_logger.log_metrics(res_dic, '/tmp/', experiment,
                                      tr_step, val_step)
#
#     # model_class.save_if_best(
#     #     hparams['tn_mask_dir'], model.module, opt, tr_step,
#     #     res_dic[back_loss_tr_loss_name]['mean'],
#     #     res_dic[val_loss_name]['mean'], val_loss_name.replace("_", ""))
    for loss_name in res_dic:
        res_dic[loss_name]['acc'] = []
    pprint(res_dic)
