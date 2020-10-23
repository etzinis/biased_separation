"""!
@brief Library for saving metrics per sample and per epoch

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import numpy as np


def log_metrics(metrics_dict, dirpath, experiment, tr_step, val_step):
    """Logs the accumulative individual results from a dictionary of metrics

    Args:
        metrics_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment: Experiment object from cometml, used for prefix.
        dirpath:  An absolute path for saving the metrics into
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation
    """
    prefix = experiment.get_key()
    # experiment.log_asset_folder('metrics', step=None, log_file_name=False,
    #                             recursive=False)

    for metric_name, metric_data in metrics_dict.items():
        this_metric_folder = os.path.join(dirpath, prefix+metric_name)
        if not os.path.exists(this_metric_folder):
            print("Creating non-existing metric log directory... {}"
                  "".format(this_metric_folder))
            os.makedirs(this_metric_folder)

        if isinstance(metric_data, dict):
            values = metric_data['acc']
        else:
            values = metric_data
        values = np.array(values)
        if 'tr' in metric_name:
            filename = 'epoch_{}.npy'.format(tr_step)
            with experiment.train():
                actual_save_path = os.path.join(this_metric_folder, filename)
                np.save(actual_save_path, values)
                experiment.log_asset(actual_save_path,
                                     file_name='metrics/'+metric_name,
                                     overwrite=True, copy_to_tmp=True,
                                     step=tr_step, metadata=None)
        else:
            filename = 'epoch_{}.npy'.format(val_step)
            with experiment.validate():
                actual_save_path = os.path.join(this_metric_folder, filename)
                np.save(actual_save_path, values)
                experiment.log_asset(actual_save_path,
                                     file_name='metrics/' + metric_name,
                                     overwrite=False, copy_to_tmp=True,
                                     step=val_step, metadata=None)

