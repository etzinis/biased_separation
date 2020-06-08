"""!
@brief Experiment Argument Parser

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Experiment Argument Parser')
    # ===============================================
    # Datasets arguments
    parser.add_argument("--train", type=str, nargs='+',
                        help="Training dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX'])
    parser.add_argument("--val", type=str, nargs='+',
                        help="Validation dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX'])
    parser.add_argument("--test", type=str, nargs='+',
                        help="Test dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX'])
    parser.add_argument("--train_val", type=str, nargs='+',
                        help="Validation on the training data",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX'])
    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training 
                                samples to this number.""", default=None)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation 
                                samples to this number.""", default=None)
    # ===============================================
    # Training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch. 
                                Warning: Cannot be less than the number of 
                                the validation samples""", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the 
                            experiment should run""", default=50)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-2)
    parser.add_argument("--divide_lr_by", type=float,
                        help="""The factor that the learning rate 
                            would be divided by""", default=1.)
    parser.add_argument("--reduce_lr_every", type=float,
                        help="""Reduce learning rate every how many 
                            training epochs? 0 means that the learning 
                            rate is not going to be divided by the 
                            specified factor.""",
                        default=0)
    parser.add_argument("-fs", type=float,
                        help="""Sampling rate of the audio.""", default=8000.)
    # ===============================================
    # CometML experiment configuration arguments
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml 
                            experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="yolo_experiment")
    # ===============================================
    # Device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                            available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for 
                                        loading the data, etc.""", default=4)
    # ===============================================
    # Local experiment logging
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for logging experiment's audio.""",
                        default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="""Path for logging metrics.""",
                        default=None)
    # ===============================================
    # Separation model (SuDO-RM-RF) params
    parser.add_argument("--out_channels", type=int,
                        help="The number of channels of the internal "
                             "representation outside the U-Blocks.",
                        default=128)
    parser.add_argument("--in_channels", type=int,
                        help="The number of channels of the internal "
                             "representation inside the U-Blocks.",
                        default=512)
    parser.add_argument("--num_blocks", type=int,
                        help="Number of the successive U-Blocks.",
                        default=16)
    parser.add_argument("--upsampling_depth", type=int,
                        help="Number of successive upsamplings and "
                             "effectively downsampling inside each U-Block. "
                             "The aggregation of all scales is performed by "
                             "addition.",
                        default=3)
    parser.add_argument("--enc_kernel_size", type=int,
                        help="The width of the encoder and decoder kernels.",
                        default=21)
    parser.add_argument("--enc_num_basis", type=int,
                        help="Number of the encoded basis representations.",
                        default=512)
    return parser.parse_args()
