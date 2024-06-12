import argparse
import numpy as np

# ----------------------------------------------------------------------------
def updateWeightsPath(args):
    # if args.multiple_datasets:
    #     init_name_w = 'weights_dbAll'
    # else:
    init_name_w = 'weights_db' + str(args.dbs)

    # if args.multiple_set_samples:
    #     set_samples_txt = '_multipleSetSamples'
    # else:
    set_samples_txt = '__samples' + str(args.samples)

    path = init_name_w + set_samples_txt \
            + '__pos' + str(args.pos) \
            + '__neg' + str(args.neg) \
            + '__e' + str(args.epochs) \
            + '__b' + str(args.batch) \
            + '__dim' + str(args.dim) \
            + ('__augtr' + str(args.augtr) if args.augtr else '' ) \
            + '__experiment' + str(args.log_table) \
            + '__size' + str(args.train_size) \
            + ('__pretrainedW' + args.pretrained_weights if args.pretrained_weights else '')  \
            + '__freeze' + (str(args.freeze) if args.freeze else "0")  \
            + ('__partition' + str(args.partition) if args.partition > -1 else "")  \
            + ('__from' + str(args.init_dbs) if args.init_dbs else "")  
            
            
            # + '.h5'
            # + '_loss' + args.loss \
            # + '_opt' + args.optimizer \

    args.log_filename = path
    args.weights_filename = 'WEIGHTS/' + path #+ '.h5'

    return args

# ----------------------------------------------------------------------------
def args():
    parser = argparse.ArgumentParser(description='Siamese')


    group0 = parser.add_argument_group('Training type')
    group0.add_argument('-type',   default=None, type=str,  help='Training experiment')

    group1 = parser.add_argument_group('Input data parameters')

    group1.add_argument('-load_by_disk',   default=True, type=bool, help='Load all data directly from disk. If false, load from memory (faster).')
    
    group1.add_argument('-dbs',   default=None, type=str, help='Comma separated list of BD names to load. Set None to load all.')
    group1.add_argument('-init_dbs',   default=None, type=str, help='Pre-trained dataset to fine-tuning.')    
    group1.add_argument('-tl_based_tgt_dbs',   default=None, type=str, help='Target dataset to train on.')    
    group1.add_argument('-init_weights',   default=None, type=str, help='Path to dataset to fine-tuning. It is overwritten.')    
    group1.add_argument('-multiple_datasets',   default=None, type=str, help='Experiment with one or all datasets. Set None to load only one.')
    group1.add_argument('-pretrained_weights',   default=None,                 type=str,
                                                help='Pretrained weights as a backbone')
    group1.add_argument('-tl_weights',   default=None,                 type=str,
                                                help='Pretrained weights path to transfer learning')
    group1.add_argument('-start',   default=0,  type=int,   help='Offset to get train samples')
    group1.add_argument('-cv_partitions',   default=5,  type=int,   help='Number of training partitions to be trained on')
    group1.add_argument('-partition',   default=-1,  type=int,   help='Number of training partition trained')
    group1.add_argument('-freeze',   default=None,  type=str,   help='Number of epochs to freeze layers')
    group1.add_argument('-dist_batch_imbalance',   default=None,  type=str,   help='Use imbalance of batch')
    # group1.add_argument('-dist_imbalance',   default=None,  type=str,   help='Use imbalance of dist')

    group1.add_argument('-samples',   default=5,  dest='samples', type=int,   help='Number remaining train samples on minority class in dataset')
    group1.add_argument('-multSamples',   default=1,  dest='multSamples', type=int,   help='Multiply the original set of samples')
    group1.add_argument('-multiple_set_samples',   default=None, type=str, help='Experiment with one or all samples for min class. Set None to load only one.')
    group1.add_argument('-limit_train_size',   default=100,  dest='limit_train_size', type=int,   help='Number remaining train samples on majority dataset')
    group1.add_argument('-train_size',   default=100,  dest='train_size', type=int,   help='Number of train samples per class')
    # group1.add_argument('-size',   default=-1, type=int, help='Scale to this size. -1 to use default size.')

    group3 = parser.add_argument_group('Training parameters')
    group3.add_argument('-e',          default=2,    type=int,         dest='epochs',         help='Number of epochs')
    group3.add_argument('-b',          default=16,     type=int,       dest='batch',            help='Batch size')

    group3.add_argument('-dim',   default=32,  type=int,   help='Embedding dimensions')
    group3.add_argument('-opt',   default='sgc', dest='optimizer', type=str, help='Optimizer')
    group3.add_argument('-w_loss',   default=None, type=str, help='Use or not of weighted loss')
    group3.add_argument('-loss',   default='a', type=str, choices=['a', 'b', 'c'], help='Loss function')
    group3.add_argument('-pos',   default=1,  type=int,   help='Number of positive pairs')
    group3.add_argument('-neg',   default=5,  type=int,   help='Number of negative pairs')
    group1.add_argument('-multiple_neg',   default=None, type=str, help='Experiment with different set of pairings.')

    group3.add_argument('-augtr',  default=0,  type=int,   help='Use train data augmentation')

    group3.add_argument('--load',  action='store_true',               help='Load weights.')
    group3.add_argument('-gpu',    default='0',       type=str,         help='GPU')
    group3.add_argument('-save_weights',    default=None,       type=str,         help='Store weights of execution model')
    group3.add_argument('-loss_curves',    default=True,       type=str,         help='Print loss curves or not to speed execution')
    group3.add_argument('-lossCurvesTest',    default=None,       type=str,         help='Print test loss curves or not to speed execution')
    group3.add_argument('-log_table',    default=None,       type=str,         help='Name of file of table results located in log/tableResults')
    group3.add_argument('-reload_data',    default=None,       type=str,         help='Name of file of table results located in log/tableResults')
    group3.add_argument('-evaluateHistOnly',    default=False,       type=str,         help='Evaluate only by hist or by the others')
    
    args = parser.parse_args()

    args = updateWeightsPath(args)

    print('CONFIG:\n -', str(args).replace('Namespace(','').replace(')','').replace(', ', '\n - '))

    return args


def saveConfig(config):
    return config.augtr,config.epochs, config.limit_train_size, config.log_filename, config.log_table, config.multiple_datasets, \
                config.multiple_neg, config.multiple_set_samples, config.neg, config.pos, config.samples, config.train_size,\
                config.type, config.w_loss, config.start, config.weights_filename, config.pretrained_weights, \
                config.freeze
    
def storeConfig(config, oldCData):
    config.augtr,config.epochs, config.limit_train_size, config.log_filename, config.log_table, config.multiple_datasets, \
                config.multiple_neg, config.multiple_set_samples, config.neg, config.pos, config.samples, config.train_size,\
                config.type, config.w_loss, config.start, config.weights_filename, config.pretrained_weights, \
                config.freeze = oldCData
    return config


def loadModelNp(model, path):
    weight = np.load(path + '.npy', allow_pickle=True)
    model.set_weights(np.array(weight, dtype=object))

def saveModelNp(model, path):
    np.save(path, np.array(model.get_weights(), dtype=object))
