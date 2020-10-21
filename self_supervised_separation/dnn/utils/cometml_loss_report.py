"""!
@brief Library for experiment loss functionality

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def create_new_scatterplot(x_name, x_data, y_name, y_data, prefix=''):
    if len(x_data) != len(y_data):
        import pdb; pdb.set_trace()
    plt.figure()
    plt.scatter(x_data, y_data, alpha=0.5)
    plt.ylabel(y_name, fontsize=24)
    plt.xlabel(x_name, fontsize=24)
    x_lim = plt.xlim()
    if y_name.endswith('i'):
        plt.plot(x_lim, [0, 0], 'k-', color='r')
    else:
        plt.plot(x_lim, x_lim, 'k-', color='r')
    figpath = os.path.join('/tmp', prefix+x_name+y_name+'.png')
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    return figpath


def report_scatterplots(scatterplots_list, experiment, tr_step, val_step):
    """Only reports metrics"""
    exp_id = experiment.get_key()
    # Subsample for better visualization.
    subsampling_factors = [2, 4, 8]
    for factor in subsampling_factors:
        prefix = exp_id + 'subsample' + str(factor)
        for x_data, y_data in scatterplots_list:
            x_name, x_data_in_list =  x_data
            y_name, y_data_in_list =  y_data
            x_data_in_numpy = np.array(x_data_in_list)
            n_samples = int(x_data_in_numpy.shape[0] / factor)
            x_data_in_numpy = x_data_in_numpy[::factor]
            y_data_in_numpy = np.array(y_data_in_list)
            y_data_in_numpy = y_data_in_numpy[::factor]

            if 'val' in x_name or 'test' in x_name:
                with experiment.validate():
                    path = create_new_scatterplot(x_name, x_data_in_numpy,
                                                  y_name, y_data_in_numpy,
                                                  prefix=prefix)
                    experiment.log_image(
                        path, name=y_name+'n_samples_'+str(n_samples),
                        overwrite=False,
                        image_format="png", image_scale=1.0, image_shape=None,
                        image_colormap=None, image_minmax=None,
                        image_channels="last", copy_to_tmp=True, step=val_step)
            elif 'tr' in x_name:
                with experiment.train():
                    path = create_new_scatterplot(x_name, x_data_in_numpy,
                                                  y_name, y_data_in_numpy,
                                                  prefix=prefix)
                    experiment.log_image(
                        path, name=y_name+'n_samples_'+str(n_samples),
                        overwrite=False,
                        image_format="png", image_scale=1.0, image_shape=None,
                        image_colormap=None, image_minmax=None,
                        image_channels="last", copy_to_tmp=True, step=tr_step)
            else:
                raise ValueError("tr or val or test must be in metric name <{}>."
                                 "".format(x_name))


def report_histograms(histograms_dict, experiment, tr_step, val_step):
    for h_name in histograms_dict:
        if 'val' in h_name or 'test' in h_name:
            with experiment.validate():
                experiment.log_histogram_3d(
                    histograms_dict[h_name], name=h_name, step=val_step)
        elif 'tr' in h_name:
            with experiment.train():
                experiment.log_histogram_3d(
                    histograms_dict[h_name], name=h_name, step=tr_step)
        else:
            raise ValueError("tr or val or test must be in metric name <{}>."
                             "".format(h_name))


def report_losses_mean_and_std_combinations(
        losses_dict, mask_dic, combs, experiment, tr_step, val_step):
    rename_dic = ['speech', 'env_sound']

    for l_name in losses_dict:
        values_np = np.array(losses_dict[l_name]['acc'])
        if l_name in mask_dic:
            mask_np = np.array(mask_dic[l_name])
        else:
            continue
        values_reshaped = values_np.reshape(-1, 2)
        mask_np_reshaped = mask_np.reshape(-1, 2)

        for comb in combs:
            new_name = '+'.join([rename_dic[c] for c in comb])
            values = values_reshaped.copy()
            if len(comb) > 1:
                values = values[
                    (mask_np_reshaped[:, comb[0]]==comb[0]) *
                    (mask_np_reshaped[:, comb[1]]==comb[1])]
            else:
                values = values_np[mask_np==comb[0]]

            mean_metric = np.mean(values)
            std_metric = np.std(values)

            if 'val' in l_name or 'test' in l_name:
                actual_name = l_name.replace('val_', '') + '_' + new_name
                with experiment.validate():
                    experiment.log_metric(actual_name + '_mean',
                                          mean_metric,
                                          step=val_step)
                    experiment.log_metric(actual_name + '_std',
                                          std_metric,
                                          step=val_step)
            elif 'tr' in l_name:
                actual_name = l_name.replace('tr_', '') + '_' + new_name
                with experiment.train():
                    experiment.log_metric(actual_name + '_mean',
                                          mean_metric,
                                          step=tr_step)
                    experiment.log_metric(actual_name + '_std',
                                          std_metric,
                                          step=tr_step)

            else:
                raise ValueError("tr or val or test must be in metric name <{}>."
                                 "".format(l_name))


def report_losses_mean_and_std(losses_dict, experiment, tr_step, val_step):
    """Wrapper for cometml loss report functionality.

    Reports the mean and the std of each loss by inferring the train and the
    val string and it assigns it accordingly.

    Args:
        losses_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment:  A cometml experiment object
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation

    Returns:
        The updated losses_dict with the current mean and std
    """

    for l_name in losses_dict:
        values = losses_dict[l_name]['acc']
        mean_metric = np.mean(values)
        std_metric = np.std(values)

        if 'val' in l_name or 'test' in l_name:
            actual_name = l_name.replace('val_', '')
            with experiment.validate():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=val_step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=val_step)
        elif 'tr' in l_name:
            actual_name = l_name.replace('tr_', '')
            with experiment.train():
                experiment.log_metric(actual_name + '_mean',
                                      mean_metric,
                                      step=tr_step)
                experiment.log_metric(actual_name + '_std',
                                      std_metric,
                                      step=tr_step)

        else:
            raise ValueError("tr or val or test must be in metric name <{}>."
                             "".format(l_name))

        losses_dict[l_name]['mean'] = mean_metric
        losses_dict[l_name]['std'] = std_metric

    return losses_dict
