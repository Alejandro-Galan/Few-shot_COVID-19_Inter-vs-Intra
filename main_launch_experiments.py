import utils.util as util
import numpy as np
import utils.utilLoad as utilLoad
import pickle, os, copy, gc
from utils import utilGeneralSiameseCovid as utilG
from utils import utilConfig
from pympler import asizeof
import warnings, tensorflow

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def init(config, partition = True, basePath = None):
    util.init() # Reproducibility

    # In cases dataset has same parameters than other previous iteration, no need to reload 
    # datasetIndex = "Db" + str(config.dbs) + "_S" + str(config.samples) + \
    #                "_Off" + str(config.start)
    # if datasetIndex in prevDatasets: 
    #     print("Loading dataset from storage")
    #     dataset = prevDatasets[datasetIndex]
    #     input_shape, num_labels = None, None
    # else:
    #     # Mix datasets for dann approach
    #     dataset, input_shape, num_labels = utilLoad.load(config, partition == 0)
    #     prevDatasets[datasetIndex] = dataset

    source_path = basePath + "dataset" + str(config.dbs) + "/samples" + str(config.samples) + \
            "/pos" + str(config.pos) + "_neg" + str(config.neg) + "/partition" + str(partition) + "_training.pkl"

    source_dataset_path = basePath + "dataset" + str(config.dbs) + ".pkl"
    config.partition = partition
    config = utilConfig.updateWeightsPath(config)
    if os.path.exists(source_dataset_path):
        mat_te = loadDict(source_dataset_path)



    if not os.path.exists(source_path):
        # Store all mat_tr for datasets if not exist
        # Only one in case of config
        datasets_nums = [int(config.dbs)] # Source dataset

        if config.multiple_datasets or config.tl_based_tgt_dbs:
            datasets_nums = [(int(config.dbs) - 2) % 3 + 1, (int(config.dbs) - 3) % 3 + 1, (int(config.dbs) - 1) % 3 + 1] # We want the source dataset as the last one
        for dataset_num in datasets_nums:
            path_folder = basePath + "dataset" + str(dataset_num) + "/samples" + str(config.samples) + "/pos" + str(config.pos) + "_neg" + str(config.neg) + "/"
            path_target = path_folder + "partition" + str(partition) + "_training.pkl"
            if not os.path.exists(path_target):
                # if config.tl_based_tgt_dbs and int(config.tl_based_tgt_dbs) != dataset_num:
                #     continue

                config_tgt = copy.deepcopy(config)
                config_tgt.dbs = dataset_num
                
                dataset_tgt, input_shape, num_labels = utilLoad.load(config_tgt, partition == 0)

                mat_tr, mat_te, mat_val, gen_tr, gen_te, config_tgt = utilG.load_data_siamese(dataset_tgt, config_tgt, partition == 0)       

                os.system("mkdir -p " + path_folder)
                with open(path_target, 'wb') as fp:
                    pickle.dump({"mat_tr": mat_tr, "mat_val": mat_val, "gen_tr": gen_tr, 
                                "gen_te": gen_te, "config": config_tgt}, fp)

                path_dataset_test_tgt = basePath + "dataset" + str(config_tgt.dbs) + ".pkl"
                if not os.path.exists(path_dataset_test_tgt):
                    with open(path_dataset_test_tgt, 'wb') as fp: # Same test for all dataset
                        pickle.dump(mat_te, fp) # mat_te

    # It was already loaded before or loaded now
    data = loadDict(source_path)
    # Data not wanted to be overwritten
    oldCData = utilConfig.saveConfig(config) 
    mat_tr, mat_val, gen_tr, gen_te, config = data["mat_tr"], data["mat_val"], data["gen_tr"], data["gen_te"], data["config"]    
    config = utilConfig.storeConfig(config, oldCData)
    # Some machines are found limitation over big datasets, only few data for training is being used
    # del dataset

    return mat_tr, mat_te, mat_val, gen_tr, gen_te, None, None, config


# Global variables to clean
LIST_DELETE_G = ["initData", "mat_tr", "mat_te", "mat_val", "emb_model"]

def clean_RAM(printResults=False, list_delete_g=LIST_DELETE_G):
    tensorflow.compat.v1.reset_default_graph()
    
    for g in list_delete_g:
        if g in globals().keys():
            del globals()[g]

    gc.collect()

    if printResults:
        all_variables = {}
        for name in globals().keys(): 
            if name.startswith('_'):
                continue
            var = globals()[name]
            size = asizeof.asizeof(var)
            all_variables[int(size/1024)] = name
            # print(name, asizeof.asizeof(var))
            # print(len(globals()[name]))

            sorted_vars = {k: all_variables[k] for k in sorted(all_variables)}
            print(sorted_vars)

# Base experiment using only siamese structure. (Init or not init weights)
def siamese_base_experiment(config, initData, pretrained_weights):
    mat_tr, mat_te, mat_val, gen_tr, gen_te, _, _ = initData

    clean_RAM(printResults=False, list_delete_g=["initData", "emb_model"])

    print('Train siamese base model...')
    emb_model = utilG.train_model(None, None, gen_tr, mat_tr, mat_te, mat_val, config, pretrained_weights)

    return emb_model, mat_tr, mat_te, config

# Return path of the pre-trained dataset for each dataset
def getInitWeightsPath(dataset_number, tl_weights):
    path1 = "TRANSFER_LEARNING_WEIGHTS/weights_db_"
    path2 = "_cnn_tl_samples-1__pos1__neg1__e50__b8__dim32__experimentcnn-Bal-InitW-pos1_neg1-trainSize-1-SAVE_WEIGHTS__size-1__pretrainedWimagenet"
    if dataset_number not in ["1", "2", "3"]:
        raise Exception('Unknown dataset name or number ' + dataset_number)
    if tl_weights:
        path2 = tl_weights
    else:
        path2 = str(dataset_number) + path2
    return path1 + path2

def evalMeanPartitions(part_d):
    # Get all partitions and do the average
    partitions = None

    part_results = {'partition':[],'hist':[],'svr':[],'knn':[],'rf':[]}
    for d in part_d:
        eval_partition = utilG.run_evaluation(d["emb_model"], d["mat_tr"], 
                            d["mat_te"], d["config"])

        part_results['partition'].append(d['partition'])
        part_results['hist'].append(eval_partition['hist'])
        part_results['svr'].append(eval_partition['svr'])
        part_results['knn'].append(eval_partition['knn'])
        part_results['rf'].append(eval_partition['rf']) 
        if not partitions:
            partitions = eval_partition 
        else:
            for metric, p in partitions.items():
                partitions[metric] = [a + b for a,b in zip(p, eval_partition[metric])]

    for metric, p in partitions.items():
        partitions[metric] = [a / len(part_d) for a in p]
    print("RESULTS TO AVERAGE BETWEEN PARTITIONS", part_results, "MEAN", np.mean(part_results['hist']), partitions["hist"][3])
    return partitions, part_results


def loadDict(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def initLogTable(config):
    if config.log_table:
        if not config.tl_based_tgt_dbs:
            os.system("rm -f log/resultsTable/" + config.log_table + ".txt")
        if not os.path.exists("log/resultsTable/" + config.log_table + ".txt"):
            text_file = open("log/resultsTable/" + config.log_table + ".txt", 'w')
            titleTable = ("{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}\n").format('Partition','S.Dataset', 'T.Dataset', 'Evaluation', 
                                            'N.Pos/Neg', 'N.Sampl', 'Aug', 'Experiment', 'Train.Size',
                                            'Accuracy', 'Precision', 'Recall', 'F1', 'Init DB')
            text_file.write(titleTable)
            text_file.close()   


# Dump a line per group of partitions
def printLineTableResults(base_path, config):
    if not config.log_table:
        return 
    path_configs =  "/samples" + str(config.samples) + \
                  "/pos" + str(config.pos) + "_neg" + str(config.neg)
    source_path = base_path + "dataset" + str(config.dbs) + path_configs

    all_metrics = []
    for dataset_path in os.listdir(base_path):
        if dataset_path.endswith(".pkl"):
            continue
        dataset_path_target = base_path + dataset_path
        dataset_num_tg = int(dataset_path.split("dataset")[1].split("/")[0])
        
        if config.tl_based_tgt_dbs:
            if dataset_num_tg != int(config.tl_based_tgt_dbs):
                continue

        mat_te_target = loadDict(dataset_path_target + ".pkl")
            
        part_d_set = []
        for partition_path in os.listdir(source_path):
            if partition_path.endswith("_training.pkl"):
                part_path_crop = partition_path.split("_training.pkl")[0]
                partition_path_source = source_path + "/" +  part_path_crop
                partition_path_target = dataset_path_target + path_configs + "/" + part_path_crop

                part_d = loadDict(partition_path_target + "_training.pkl")
                part_d['config'] = loadDict(partition_path_source + "_training.pkl")['config']
                part_d['emb_model'] = loadDict(partition_path_source + "_model.pkl")['emb_model']

                
                part_d["mat_te"] = mat_te_target
                part_d['partition'] = int(partition_path.split('_')[0].split('partition')[1])
                part_d_set.append(part_d)

        partitions, part_results = evalMeanPartitions(part_d_set)

        for type_eval, part in part_results.items():
            if type_eval == 'partition':
                continue
            new_row_index = [config.dbs, dataset_num_tg, str(config.pos) + "/" + str(config.neg), config.samples] 
            new_row_data  = {'partition': part_results['partition'], 'metric': part_results[type_eval], 'type_eval': type_eval }
            all_metrics.append([new_row_index, new_row_data])
    
    os.system("rm -f " + partition_path_source + "_model.pkl") # The higher cost in terms of disk space
    
    datasets = {"1": "Chest-Git", "2": "Pad-BIM", "3": "BIMCV-COVID"}
    lines = ""
    init_dbs = datasets[str(config.init_dbs)] if config.init_dbs else 0
    for item in all_metrics:
        partition_num = item[1]['partition']
        metrics_sample = item[1]['metric']
        index = item[0]     
        type_eval = item[1]['type_eval']
        for i, metrics in enumerate(metrics_sample):
            lineTable = ("{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}__{:<10}").format(
                    partition_num[i], datasets[str(index[0])], 
                    datasets[str(index[1])], type_eval, index[2], index[3], config.augtr, config.log_table, config.train_size, 
                    round(metrics[0], 4), round(metrics[1], 4), round(metrics[2], 4), round(metrics[3] * 100, 4), init_dbs)
            lines += lineTable + "\n"
    
    text_file = open("log/resultsTable/" + config.log_table + ".txt", "a")
    text_file.write(lines)
    text_file.close()
    print("Printed in table", "log/resultsTable/" + config.log_table + ".txt")





# Create folder if not exists
def createFolder(folderPath):
    os.system("mkdir -p " + folderPath)
    return folderPath

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":

    config = utilConfig.args()

    ## Multiple datasets
    # To be executed on one or multiple datasets
    if config.multiple_datasets:
        datasets_nums = [1, 2, 3]
    else:
        datasets_nums = [config.dbs]

    ## Multiple set of samples for minority class 
    if config.multiple_set_samples:
        all_samples_min = [1, 100, 50, 10] 
        all_samples_min = [el * config.multSamples for el in all_samples_min]
        # Test of original distribution. Exp with dist, dist + batch, dist + batch + w_loss
        # if config.type.startswith("original_Imbalanced_distribution"):
        #     all_samples_min = [1.01, 11.11, 53.85] 
    else:
        all_samples_min = [config.samples * config.multSamples]

    all_neg_IMB = [[149.5, 1], [13, 1], [2.275, 1]]            
    ## Kind of imbalance test. Different pairings
    if config.multiple_neg:
        arr_pos = [4, 3, 2, 3, 4, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        arr_neg = [1, 1, 1, 2, 3, 1, 4, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # arr_pos = [3, 1]
        # arr_neg = [1, 1]        
        all_neg = [[a,b] for a,b in zip(arr_pos, arr_neg)] 
    else:
        all_neg = [[config.pos, config.neg]]



    # Model, data and config needed to run evaluation
    # eval_data = {}
    base_path = "savedModels/"
    
    # Comment only when sured. Some args like distribution are not controlled
    # if config.reload_data: # Difficult to mantain
    if False:
        print("\nLoading partition data from disk, remember to erase it if change on datasets\n")
    else:
        os.system("rm -f -r " + base_path)
    
    createFolder(base_path)
    initLogTable(config)
    config.customPairing = False
    for dataset_number in datasets_nums:
        ds_dump_file = createFolder(base_path + "dataset" + str(dataset_number) + "/")
        # Overwrite the datset to be analyzed on
        config.dbs = dataset_number
        # eval_dataset = {}
        for num_samples in all_samples_min:
            samples_dump_file = createFolder(ds_dump_file + "samples" + str(num_samples) + "/")

            # print("\nNum.Negatives", num_neg)
            config.samples = num_samples
            # eval_negs = {}
            for num_pos_neg in all_neg:
                config.pos = num_pos_neg[0]
                config.neg = num_pos_neg[1]
                
                if config.dist_batch_imbalance:
                    config.pos = -1
                    config.neg = -1
                    
                    # config.pos = all_neg_IMB[int(config.dbs) - 1][0]
                    # config.neg = all_neg_IMB[int(config.dbs) - 1][1]


                neg_dump_file = createFolder(samples_dump_file + "pos" + str(config.pos) 
                                              + "_neg" + str(config.neg) + "/")
                print("\nNum.Samples:", num_samples, "\nNum.Positives:", config.pos, "\nNum.Negatives:", config.neg, "\nDataset:", dataset_number,"\n")
                
                # One for each partition of data, kind of cross val only for train
                config.base_offset  = 100
                config.start = 0
                # eval_parts = {}
                for partition in range(int(config.cv_partitions)):
                    clean_RAM(printResults=False)

                    part_dump_file = neg_dump_file + "partition" + str(partition) 

                    config.start = config.base_offset * partition

                    initData = init(config, partition = partition, basePath=base_path)
                    config = initData[-1]
                    ## Change this input depending on the experiment
                    # From scratch

                    if config.init_dbs:
                        # If not assigning pretrained_weights, no init by default 
                        config.init_weights = getInitWeightsPath(config.init_dbs, config.tl_weights)
                    else:
                        config = utilConfig.updateWeightsPath(config)


                    if config.type == "scratch":
                        config = utilConfig.updateWeightsPath(config)
                        emb_model, mat_tr, mat_te, config = siamese_base_experiment(config, initData[:-1], config.pretrained_weights)      
                    
                    # transfer-learning is just selecting init dbs in the config
                    elif config.type == "transfer-learning":
                        config = utilConfig.updateWeightsPath(config)                            
                        emb_model, mat_tr, mat_te, config = siamese_base_experiment(config, initData[:-1], config.pretrained_weights)
                        
                    elif config.type == "data_aug":
                        # If not assigning pretrained_weights, no init by default 
                        if config.augtr < 1: # It should be provided by parameters
                            raise Exception("\n\n\n\n Experiment", config.type, "should indicate higher -augtr than", config.augtr)
                        emb_model, mat_tr, mat_te, config = siamese_base_experiment(config, initData[:-1], config.pretrained_weights)

                    elif config.type == "cnn":
                        print('Train CNN...')
                        mat_tr, mat_te, mat_val, gen_tr, gen_te, _, _ = initData[:-1]
                        model = utilG.train_cnn(None, None, gen_tr, mat_tr, mat_te, mat_val, config, "imagenet")

                    else:
                        raise Exception('Unknown experiment name' + config.type)
                    
                    if config.type == "cnn":
                        with open(part_dump_file + "_model.pkl", 'wb') as fp:
                            pickle.dump({"emb_model": model}, fp)
                    else: 
                        with open(part_dump_file + "_model.pkl", 'wb') as fp:
                            pickle.dump({"emb_model": emb_model}, fp)
                    
                    clean_RAM(printResults=False)

                printLineTableResults(base_path, config)
                        

    # printResults(base_path, config)
    print("\nEnd of execution.")



