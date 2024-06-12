from utils import utilModels, utilConfig, utilLoad
import utils.util as util
from SiameseCode import utilModelSiamese, utilGeneratorPair, utilLoadSiamese, utilEvaluate
from pickle import dump
from pympler import asizeof
from sklearn.metrics import f1_score
from utils import utilGeneralSiameseCovid as utilG
import numpy as np
import sys, os, gc


os.system("export TF_CPP_MIN_LOG_LEVEL=2")


# Global variables to clean
LIST_DELETE_G = ["initData", "mat_tr", "mat_te", "emb_model"]

def clean_RAM(printResults=False, list_delete_g=LIST_DELETE_G):
    for g in list_delete_g:
        if g in globals().keys():
            del globals()[g]


    all_variables = {}
    for name in globals().keys(): 
        if name.startswith('_'):
            continue
        try:
            var = globals()[name]
            size = asizeof.asizeof(var)
            all_variables[int(size/1024/1024)] = name
            # print(name, asizeof.asizeof(var))
            # print(len(globals()[name]))
        except:
            continue
    gc.collect()

    sorted_vars = {k: all_variables[k] for k in sorted(all_variables)}
    if printResults:
        print(sorted_vars)


def get_num_samples_from_all_data(x_tr, y_tr):
    keys = np.sort(np.unique(y_tr))
    dict_t = {key: x_tr[y_tr[:, int(key)] == 1.0] for key in keys}
    train_size = len(dict_t[0]) / 4 * 3
    eval_size  = len(dict_t[1]) / 4 * 3
    return {0:int(train_size), 1:int(eval_size)}
#------------------------------------------------------------------------------
def get_m_siamese(dataset, n_tr, start=0, config=None):
    x_tr, y_tr = dataset['x_train'], dataset['y_train']
    x_te, y_te = dataset['x_test'],  dataset['y_test']
    
    # print(' - x_train:', x_tr.shape)
    # print(' - y_train:', y_tr.shape)
    # print(' - x_test:', x_te.shape)
    # print(' - y_test:', y_te.shape)

    # x_tr, y_tr = utilLoadSiamese.__normalize(x_tr, y_tr)
    # x_te, y_te = utilLoadSiamese.__normalize(x_te, y_te)
    
    num_samples_classes={0:config.train_size,1:config.samples}
    if config.train_size == -1:
        num_samples_classes=get_num_samples_from_all_data(x_tr, y_tr)

    mat_tr, weight_bal = utilLoadSiamese.__transform_to_matrix(x_tr, y_tr, n_tr, start, num_samples_classes=num_samples_classes)
    mat_val, _ = utilLoadSiamese.__transform_to_matrix_val(x_tr, y_tr, n_tr, start, num_samples_classes=num_samples_classes)
    mat_te, _ = utilLoadSiamese.__transform_to_matrix(x_te, y_te)

    # print(' - mat_tr:', mat_tr.shape)

    # Moved to train
    # if aug > 0:
    #     print('# Add data augmentation...')
    #     mat_tr = utilLoadSiamese.add_data_augmentation(mat_tr, aug)
    #     print(' - Aug tr shape:', mat_tr.shape)

    return mat_tr, mat_te, mat_val, weight_bal


#------------------------------------------------------------------------------
def load_data_siamese(dataset, config, verbose):
    mat_tr, mat_te, mat_val, weight_bal = get_m_siamese(dataset, config.train_size, 0, config=config)

    config.weight_bal = weight_bal
    # mat_me = np.zeros((mat_tr.shape[0], config.dim))     # Para el modelo con memoria...

    if verbose:
        print("mat_tr", mat_tr.shape[0], mat_tr.shape[1], mat_tr.shape[2:])
        print("mat_val", mat_val.shape[0], mat_val.shape[1], mat_val.shape[2:])
        print("mat_te", mat_te.shape[0], mat_te.shape[1], mat_te.shape[2:])

    distribution = None
    # Batch imbalance
    if config.dist_batch_imbalance:
        distribution = config.weight_bal
    gen_tr = utilGeneratorPair.pairgen(mat_tr, config.pos, config.neg, config.batch, aug = config.augtr, train = True, distribution=distribution, unbalanced=config.dist_batch_imbalance, config=config)
    gen_te = utilGeneratorPair.pairgen(mat_te, config.pos, config.neg, config.batch, config=config)


    return mat_tr, mat_te, mat_val, gen_tr, gen_te, config

# Assign a custom weight to each class
def assignWeightsSamples(l_i, weight_bal):
    vector = []

    correspondence = []
    neg, posS, posE = 0,0,0
    # Each label
    for l in l_i:
        # vector.append(weight_bal[l[0]] * weight_bal[l[1]])

        
        # if negative, the weight is the mean
        if l[0] != l[1]:
            vector.append(np.mean(list(weight_bal.values())))
            correspondence.append( {"N(_)":np.mean(list(weight_bal.values()))} )
            neg += 1
        # Pre-calculated weight of the distribution
        else:
            vector.append(weight_bal[l[0]])
            if l[0] == 0:
                correspondence.append( {"P(h)":weight_bal[l[0]]} )
                posS += 1
            elif l[0] == 1:
                correspondence.append( {"P(d)":weight_bal[l[0]]} )
                posE += 1
    
    #print(correspondence)
    #print("Negatives:", neg, "\nPositives Health:", posS, "\nPositives Disease", posE)
    return np.array(vector)

# Train on all samples possible depending on batch and remove it from the stack
def train_miniB(model, x_mult, y_mult, batch, counterSPE, limitSize, weight_bal, labels_mult, config):
    # Shuffle the data in the hole miniB to avoid the same info to lose
    p = np.random.permutation(len(y_mult))

    x_mult['i1'], x_mult['i2'], y_mult, labels_mult = x_mult['i1'][p], x_mult['i2'][p], y_mult[p], labels_mult[p]
    x_mult['i1_augm'], x_mult['i2_augm'] = x_mult['i1_augm'][p], x_mult['i2_augm'][p]

    losses = [[],[]]
    steps = int(len(y_mult) / batch) 
    for s in range(steps):
        if counterSPE >= limitSize:
            offset = (s - 1) * batch if s > 0 else 0 # The remaining samples not possible to be trained on, in case there are
            x_mult['i1'], x_mult['i2'] = x_mult['i1'][offset:], x_mult['i2'][offset:]
            x_mult['i1_augm'], x_mult['i2_augm'] = x_mult['i1_augm'][offset:], x_mult['i2_augm'][offset:]
            if losses[0]:
                print("Loss:", str(np.mean(losses[0])) + ',', str(model.metrics_names[1]) + ":", np.mean(losses[1]))

            return x_mult, y_mult[offset:], labels_mult[offset:], -1
        
        start_batch = s * batch

        x_i = {'i1':[], 'i2':[]}
        x_i['i1'] = utilLoad.read_set_img(x_mult['i1'][start_batch:start_batch + batch], x_mult['i1_augm'][start_batch:start_batch + batch], config)
        x_i['i2'] = utilLoad.read_set_img(x_mult['i2'][start_batch:start_batch + batch], x_mult['i2_augm'][start_batch:start_batch + batch], config)

        # x_i['i1'] = np.asarray(x_mult['i1'][start_batch:start_batch + batch]).astype('float32')
        # x_i['i2'] = np.asarray(x_mult['i2'][start_batch:start_batch + batch]).astype('float32')
        y_i = y_mult[start_batch:start_batch + batch]
        l_i = labels_mult[start_batch:start_batch + batch]

        if weight_bal:
            weight_batch_bal = assignWeightsSamples(l_i, weight_bal)
            l = model.train_on_batch(x_i, y_i, sample_weight=weight_batch_bal )
        else:
            l = model.train_on_batch(x_i, y_i)
        counterSPE += batch
        losses[0].append(l[0])
        losses[1].append(l[1])
        
    # loss += l[0] # 0 loss, 1 pair accuracy
    print("Loss:", str(np.mean(l[0])) + ',', str(model.metrics_names[1]) + ":", np.mean(l[1]))
    
    offset = steps * batch # The remaining samples not possible to be trained on, in case there are
    x_mult['i1'], x_mult['i2'] = x_mult['i1'][offset:], x_mult['i2'][offset:]
    x_mult['i1_augm'], x_mult['i2_augm'] = x_mult['i1_augm'][offset:], x_mult['i2_augm'][offset:]
    return x_mult, y_mult[offset:], labels_mult[offset:], counterSPE


def freezeLayers(model, freeze):
    for layer in model.layers:
        if layer.name == 'model':
            change_train = True
            for sub_layer in layer.layers:
                # print(sub_layer.name, change_train)
                if change_train:
                    sub_layer.trainable = not freeze
                    if sub_layer.name == "conv5_block3_preact_bn": #"conv5_block3_2_relu":#"conv5_block3_out": #"conv5_block3_2_relu":
                        change_train = False


#------------------------------------------------------------------------------
def train_model(input_shape, nb_classes, gen_tr, mat_tr, mat_te, mat_val, config, pretrained_weights):
    print("\n\nTRAINING PARTITION: ", config.partition)
    config = utilConfig.updateWeightsPath(config)
    # if config.loss_curves:
    #     os.system("#rm -f ./log/saveLosses/*")

    # print("Input shape", input_shape)
    base_model = utilModels.base_resnet(config, pretrained_weights)

    model, emb_model = utilModelSiamese.create(base_model)
    # print(model.summary())
    # print(emb_model.summary())
    if config.freeze:
        print("Freezing")
        freezeLayers(model, freeze=True)
        utilModelSiamese.compileFreeze(model, utilG.get_loss(config), unfreeze=False)
        # freezeLayers(emb_model, freeze=True)
        # utilModelSiamese.compileFreeze(emb_model, utilG.get_loss(config), unfreeze=False)
    else:
        utilModelSiamese.compile(model, utilG.get_loss(config), config.optimizer)

    # Disposition of balance between classes
    weight_bal = config.weight_bal[1] if config.w_loss else None  
    if config.load == True:
        # model.load_weights(config.weights_filename)
        best_model, best_emb_model = model, emb_model
        utilConfig.loadModelNp(model,config.weights_filename)
        best_f1_epoch_model = [0.0, 0]
    else:
        train_loss_dict, test_loss_dict, val_loss_dict = {}, {}, {}
        best_f1_epoch_model = [0.0, 1]
        for e in range(config.epochs): # Every virtual epoch, the size of the dataset
            # clean_RAM(printResults=False,list_delete_g=[])
            # Just once. No unfreezing
            # if config.freeze and e == config.freeze: 
            #     # for layer in model.layers
            #     freezeLayers(model, freeze=False)
            #     utilModelSiamese.compileFreeze(model, utilG.get_loss(config), unfreeze=True)            
            #     # freezeLayers(emb_model, freeze=False)
            #     # utilModelSiamese.compileFreeze(emb_model, utilG.get_loss(config), unfreeze=True)            
            
            print("##########################\nEpoch {}".format(e + 1))
            loss = 0.0
            x_mult, y_mult = {'i1':[], 'i2':[], 'i1_augm':[], 'i2_augm':[]}, []
            labels_mult = []
            counterSPE = 0 # CounterSamplesPerEpoch
            for x, y, labels_covid in gen_tr:
                x_mult['i1'] = x['i1'] if len(x_mult['i1']) == 0 else np.concatenate((x_mult['i1'], x['i1']))
                x_mult['i2'] = x['i2'] if len(x_mult['i2']) == 0 else np.concatenate((x_mult['i2'], x['i2']))
                x_mult['i1_augm'] = x['i1_augm'] if len(x_mult['i1_augm']) == 0 else np.concatenate((x_mult['i1_augm'], x['i1_augm']))
                x_mult['i2_augm'] = x['i2_augm'] if len(x_mult['i2_augm']) == 0 else np.concatenate((x_mult['i2_augm'], x['i2_augm']))
                y_mult = y if len(y_mult) == 0 else np.concatenate((y_mult, y))
                labels_mult = labels_covid if len(labels_mult) == 0 else np.concatenate((labels_mult, labels_covid))
                if len(y_mult) >= config.batch: # If there is data for at least one batch
                    x_mult, y_mult, labels_mult, counterSPE = train_miniB(model, x_mult, y_mult, config.batch, 
                                                             counterSPE, limitSize=len(mat_tr[0]), weight_bal=weight_bal, 
                                                             labels_mult=np.array(labels_mult), config=config)
                if counterSPE == -1:
                    break



            f1_val = run_evaluation(emb_model, mat_tr, mat_val, config, allM = False)['hist'][3]
            print("Lr:", model.optimizer.lr.numpy())
            print("F1 score over validation set:", f1_val, "\n##########################")
            
            if f1_val > best_f1_epoch_model[0]:
                # Must be freezed before storing
                if config.freeze: 
                    # if e >= config.freeze: 
                    os.system("rm -f " + config.weights_filename)
                    utilConfig.saveModelNp(model, config.weights_filename)
                    # model.save_weights(config.weights_filename, overwrite=True)
                    # emb_model.save_weights(config.weights_filename[:-3] + "_emb_model.h5", overwrite=True)
                    best_f1_epoch_model[0] = f1_val
                    best_f1_epoch_model[1] = e + 1
                else:
                    utilConfig.saveModelNp(model, config.weights_filename)
                    best_f1_epoch_model[0] = f1_val
                    best_f1_epoch_model[1] = e + 1





            # for i in range(mat_tr.shape[0]):
            #     new_me = np.average(emb_model.predict(mat_tr[i], verbose=0), axis=0)
            #     mat_me[i] = mat_me[i] * (1.0 - e / config.epochs) + new_me * (e / config.epochs)
            sys.stdout.flush()



            ## Loss curves
            # Not print always because slows execution
            if config.partition == 0 and config.loss_curves and (e + 1) % 1 == 0:
                # print("EVAL CURVES:")
                # Evaluate on same matrix
                train_loss = run_evaluation(emb_model, mat_tr, mat_tr, config, allM = False)
                train_loss_dict[e] = train_loss['hist'][3]

                val_loss_dict[e] = f1_val

                if config.lossCurvesTest:
                    test_loss = run_evaluation(emb_model, mat_tr, mat_te, config, allM = False)
                    test_loss_dict[e] = test_loss['hist'][3]




        ## Loss Curves To file             
        if config.partition == 0 and config.loss_curves:
            with open('./log/saveLosses/train_log_' + config.log_filename + '.pkl', 'wb') as file:
                dump(train_loss_dict, file)
            if config.lossCurvesTest:
                with open('./log/saveLosses/test_log_' + config.log_filename + '.pkl', 'wb') as file:
                    dump(test_loss_dict, file)
            with open('./log/saveLosses/valid_log_' + config.log_filename + '.pkl', 'wb') as file:
                dump(val_loss_dict, file)


        # if config.save_weights:
            # model.save_weights(config.weights_filename, overwrite=True)
                        
        best_model, best_emb_model = utilModelSiamese.create(base_model)
        utilConfig.loadModelNp(best_model, config.weights_filename)
        # best_model.load_weights(config.weights_filename)
        # best_emb_model.load_weights(config.weights_filename[:-3] + "_emb_model.h5")

        print("Best model for training saved from epoch", best_f1_epoch_model[1], "with", round(best_f1_epoch_model[0], 4), "F1 score over validation set.")

    return best_emb_model

#------------------------------------------------------------------------------
def train_cnn(input_shape, nb_classes, gen_tr, mat_tr, mat_te, mat_val, config, pretrained_weights):
    print("\n\nTRAINING PARTITION: ", config.partition)
    config = utilConfig.updateWeightsPath(config)
    # if config.loss_curves:
    #     os.system("#rm -f ./log/saveLosses/*")

    # print("Input shape", input_shape)
    model = utilModels.cnn_resnet(config, pretrained_weights)

    # print(model.summary())
    # print(emb_model.summary())
    utilModels.cnn_compile(model, utilG.get_loss(config))

    # Disposition of balance between classes
    train_loss_dict, test_loss_dict, val_loss_dict = {}, {}, {}
    best_f1_epoch_model = [0.0, 1]
    for e in range(config.epochs): # Every virtual epoch, the size of the dataset
        
        print("##########################\nEpoch {}".format(e + 1))
        loss = 0.0
        x_mult = []
        labels_mult = []
        counterSPE = 0 # CounterSamplesPerEpoch
        for x, y, labels_covid in gen_tr:
            x_mult = x['i1'] if len(x_mult) == 0 else np.concatenate((x_mult, x['i1']))
            # x_mult['i2'] = x['i2'] if len(x_mult['i2']) == 0 else np.concatenate((x_mult['i2'], x['i2']))
            labels_mult = labels_covid if len(labels_mult) == 0 else np.concatenate((labels_mult, labels_covid))

            if len(x_mult) >= config.batch: # If there is data for at least one batch
                x_mult, labels_mult, counterSPE = cnn_train_miniB(model, x_mult, None, config.batch, 
                                                            counterSPE, limitSize=len(mat_tr[0]), weight_bal=None, labels_mult=np.array(labels_mult))
            if counterSPE == -1:
                break



        f1_val = cnn_run_evaluation(model, mat_tr, mat_val, config, allM = False)['hist'][3]
        print("Lr:", model.optimizer.lr.numpy())
        print("F1 score over validation set:", f1_val, "\n##########################")
        
        if f1_val > best_f1_epoch_model[0]:
            # Must be freezed before storing
            utilConfig.saveModelNp(model, config.weights_filename)
            best_f1_epoch_model[0] = f1_val
            best_f1_epoch_model[1] = e + 1

        sys.stdout.flush()


    # if config.save_weights:
        # model.save_weights(config.weights_filename, overwrite=True)
                    
    best_model = utilModels.cnn_resnet(config, pretrained_weights)
    utilConfig.loadModelNp(best_model, config.weights_filename)
    # best_model.load_weights(config.weights_filename)
    # best_emb_model.load_weights(config.weights_filename[:-3] + "_emb_model.h5")

    print("Best model for training saved from epoch", best_f1_epoch_model[1], "with", round(best_f1_epoch_model[0], 4), "F1 score over validation set.")

    return best_model

# Train on all samples possible depending on batch and remove it from the stack
def cnn_train_miniB(model, x_mult, _, batch, counterSPE, limitSize, weight_bal, labels_mult):
    # Shuffle the data in the hole miniB to avoid the same info to lose
    p = np.random.permutation(len(x_mult))

    x_mult, labels_mult = x_mult[p], labels_mult[p]

    losses = []
    steps = int(len(x_mult) / batch) 
    for s in range(steps):
        if counterSPE >= limitSize:
            offset = (s - 1) * batch if s > 0 else 0 # The remaining samples not possible to be trained on, in case there are
            x_mult = x_mult[offset:]
            # if losses:
            #     print("Loss:", str(np.mean(losses)))
            return x_mult, labels_mult[offset:], -1
        
        start_batch = s * batch

        x_i = []
        x_i = x_mult[start_batch:start_batch + batch]
        l_i = labels_mult[start_batch:start_batch + batch]
        # Using only first element
        l_i = np.array([e[0] for e in l_i], dtype=np.float32)
        l = model.train_on_batch(x_i, l_i)
        counterSPE += batch
        losses.append(l)
        
    # loss += l[0] # 0 loss, 1 pair accuracy
    # print("Loss:", str(np.mean(l)))
    
    offset = steps * batch # The remaining samples not possible to be trained on, in case there are
    
    return x_mult[offset:], labels_mult[offset:], counterSPE


#------------------------------------------------------------------------------
def get_loss(config):

    if config.loss == 'a':
        # if config.w_loss:
        #     return utilModelSiamese.get_loss_a(config.weight_bal)
        # else:
        return utilModelSiamese.get_loss_a()
    elif config.loss == 'b':
        return utilModelSiamese.get_loss_b(config.margin)
    else: # c
        return utilModelSiamese.get_loss_c()


#------------------------------------------------------------------------------
def run_evaluation(emb_model, mat_tr, mat_te, config, allM = True):
    # print('Evaluate...')
    x_tr = np.reshape(mat_tr, (-1,) + mat_tr.shape[2:])
    
    if config.load_by_disk:
        x_tr = utilLoad.read_set_img(x_tr, np.zeros(len(x_tr)), config)    

    e_tr = emb_model.predict(x_tr, verbose=0, batch_size=config.dim)
    y_tr = np.repeat(np.arange(mat_tr.shape[0]), mat_tr.shape[1])

    x_te = np.reshape(mat_te, (-1,) + mat_te.shape[2:])
    if config.load_by_disk:
        x_te = utilLoad.read_set_img(x_te, np.zeros(len(x_tr)), config)  
        
    e_te = emb_model.predict(x_te, verbose=0, batch_size=config.dim)
    y_te = np.repeat(np.arange(mat_te.shape[0]), mat_te.shape[1])

    # print(' - Train shape:', e_tr.shape, y_tr.shape)
    # print(' - Test shape:', e_te.shape, y_te.shape)

    _, metrics_hist = utilEvaluate.hist(e_tr, y_tr, e_te, y_te)
    if allM and not config.evaluateHistOnly:
        _, metrics_svr  = utilEvaluate.svr(e_tr, y_tr, e_te, y_te)
        _, metrics_knn  = utilEvaluate.knn(e_tr, y_tr, e_te, y_te)
        _, metrics_rf   = utilEvaluate.rf(e_tr, y_tr, e_te, y_te)

    if allM and not config.evaluateHistOnly:
        return {"hist": metrics_hist, "svr": metrics_svr, 
                "knn": metrics_knn,   "rf": metrics_rf}
    return {"hist": metrics_hist}

    # if args.augte == True:
    #     run_evaluation_with_augmentation(emb_model, e_tr, y_tr, x_te, e_te, y_te)

#------------------------------------------------------------------------------
def cnn_run_evaluation(model, mat_tr, mat_te, config, allM = True):
    # print('Evaluate...')
    x_tr = np.reshape(mat_tr, (-1,) + mat_tr.shape[2:])
    pred_e_tr = model.predict(x_tr, verbose=0, batch_size=config.dim)
    y_tr = np.repeat(np.arange(mat_tr.shape[0]), mat_tr.shape[1])

    x_te = np.reshape(mat_te, (-1,) + mat_te.shape[2:])
    pred_e_te = model.predict(x_te, verbose=0, batch_size=config.dim)
    y_te = np.repeat(np.arange(mat_te.shape[0]), mat_te.shape[1])

    # print(' - Train shape:', e_tr.shape, y_tr.shape)
    # print(' - Test shape:', e_te.shape, y_te.shape)

    pred_e_te = np.array((pred_e_te > 0.5) * 1, dtype=np.int64)
    f1 = f1_score(y_te, pred_e_te, average='weighted')    

    metrics_hist = [None, None, None, f1]

    return {"hist": metrics_hist}

    # if args.augte == True:
    #     run_evaluation_with_augmentation(emb_model, e_tr, y_tr, x_te, e_te, y_te)
