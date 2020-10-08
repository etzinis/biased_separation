"""!
@brief Model logger in order to be able to load the model and test it
on different data.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""
import os
import sys
import torch
import datetime
import glob2

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../../')
sys.path.insert(0, root_dir)
from end2end_unsupervised_separation.config import MODELS_DIR


class ModelStateIO(object):
    model_suffix = '.pt'
    state_dic_suffix = '.dict'

    def __init__(self,
                 model_name=None,
                 trained_dataset_path=None,
                 trained_labels=None,
                 printed_metrics_names=None):

        if ((model_name is None) or (trained_dataset_path is None) or
                (trained_labels is None)):
            raise ValueError("All Model State Saver Should be valid "
                             "for inference of the savefolder for the "
                             "pretrainedmodels")

        self.save_dir = self.encode_model_dir(model_name,
                                              trained_dataset_path,
                                              trained_labels)

        print("Saving model states at: {}".format(self.save_dir))
        if not os.path.exists(self.save_dir):
            print("Creating non-existing model states directory...")
            os.makedirs(self.save_dir)

        if not isinstance(printed_metrics_names, list):
            raise ValueError("Printed metrics should contain a list "
                             "of tracked metrics that would "
                             "characterize each saved model")

        self.state = {
            'sorted_metrics': printed_metrics_names,
            'metrics_eval': dict([(m, 0.0) for m in
                                 printed_metrics_names]),
            'trained_dataset_path': trained_dataset_path,
            'model_name': model_name,
            'trained_labels': trained_labels,
        }

        self.torch_model = None
        self.model_suffix = ModelStateIO.model_suffix
        self.state_dic_suffix = ModelStateIO.state_dic_suffix

    def update_state(self,
                     **kwargs):
        for key, value in kwargs.items():
            if key == 'model':
                self.torch_model = value
            elif key not in self.state:
                raise KeyError("Key: {} cannot be found inside "
                               "Internal state".format(key))
            else:
                self.state[key] = value

    @staticmethod
    def encode_model_substructure(model_name,
                                  trained_dataset_path,
                                  trained_labels):
        dataset_name = os.path.basename(os.path.dirname(
                       os.path.abspath(trained_dataset_path)))

        return os.path.join(model_name,
                            dataset_name,
                            trained_labels)

    def encode_model_dir(self,
                         model_name,
                         trained_dataset_path,
                         trained_labels):

        save_model_dir = os.path.join(os.path.abspath(MODELS_DIR),
                                 self.encode_model_substructure(
                                      model_name,
                                      trained_dataset_path,
                                      trained_labels))

        return save_model_dir

    def encode_model_identifier(self):
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%s")

        file_identifiers = ["{}_{}".format(m_name,
                                           round(self.state[
                                                     'metrics_eval'][
                                                     m_name], 3))
                            for m_name in self.state['sorted_metrics']]
        model_identifier = "_".join(file_identifiers + [ts])

        return model_identifier

    def decode_model_identifier(self,
                                model_identifier):
        identifiers = model_identifier.split("_")
        ts = identifiers[-1].strip(self.model_suffix)
        metrics_of_models_dic = dict([
            (m_name, float(identifiers[2*i+1]))
            for i, m_name in enumerate(self.state['sorted_metrics'])])
        return metrics_of_models_dic, ts

    def save_state(self):
        model_identifier = self.encode_model_identifier()

        file_path = os.path.join(self.save_dir,
                                 model_identifier)
        torch.save(self.state, file_path + self.state_dic_suffix)
        torch.save(self.torch_model, file_path + self.model_suffix)

    def save_state_if_among_best(self,
                                 keep_best=20,
                                 comparison_metric='sdr'):
        self.save_state()
        available_models = glob2.glob(self.save_dir
                                      + '/*' + self.model_suffix)

        if len(available_models) > keep_best:

            identifiers = [(os.path.basename(path), path)
                           for path in available_models]

            model_decoded_info = [(self.decode_model_identifier(iden),
                                   path)
                                  for (iden, path) in identifiers]

            performance_sorted_models = sorted(
                model_decoded_info,
                key=lambda x: x[0][0][comparison_metric])

            print(performance_sorted_models)

            for j in range(len(performance_sorted_models) - keep_best):

                model_path_to_remove = performance_sorted_models[j][1]
                identifier_path = model_path_to_remove.strip(
                                  self.model_suffix)
                dic_path_to_remove = identifier_path + self.state_dic_suffix
                try:
                    os.remove(model_path_to_remove)
                    os.remove(dic_path_to_remove)
                except:
                    print("Warning: Error in removing {} ..." +
                          "".format(model_path_to_remove))

    @staticmethod
    def decode_model_dir(model_path):
        abs_path = os.path.abspath(model_path)
        model_name, dataset_name, tr_labels = abs_path.split('/')[-3:]
        return model_name, dataset_name, tr_labels

    @staticmethod
    def decode_model_path(path):

        if path.startswith(MODELS_DIR):
            model_path_structure = path[len(MODELS_DIR):]
        else:
            raise IOError(
                "Cannot parse models which are not defined in "
                "config file under MODELSDIR: {}"
                "".format(MODELS_DIR))

        return model_path_structure

    @staticmethod
    def load_model_state(path):
        identifier_path = path.strip(ModelStateIO.state_dic_suffix)\
                              .strip(ModelStateIO.model_suffix)
        model_state = torch.load(identifier_path +
                                 ModelStateIO.state_dic_suffix)
        torch_model = torch.load(identifier_path +
                                 ModelStateIO.model_suffix)
        return model_state, torch_model
