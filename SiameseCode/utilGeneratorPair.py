from utils import utilLoad
import keras
import numpy as np
import math
from keras.preprocessing.image import ImageDataGenerator


#------------------------------------------------------------------------------
def apply_data_augmentation(img_orig, aug):
    assert aug > 0

    x_orig = img_orig

    if len(img_orig.shape)==3:
        x_orig = np.expand_dims(img_orig, 0)
    
    factor_augment = aug

    datagen1 = ImageDataGenerator( rotation_range=1 * factor_augment,
                                width_shift_range=0.01 * factor_augment,
                                height_shift_range=0.01 * factor_augment,
                                zoom_range=0.01 * factor_augment)
    
    # NUM_AUGS = aug
    # for r in range(NUM_AUGS):
    x_aug = next( datagen1.flow(x_orig, batch_size=1) )[0] # Expanded dims, like a batch

        # import cv2
        # cv2.imwrite("img_o.png", img_orig)
        # cv2.imwrite("img_a.png", x_aug)


    return x_aug

# Choose a decision based on 2 probabilities
def roulette(distribution):
    decision = "POSITIVE"
    random = np.random.random()
    if random < distribution[1]:
        decision = "NEGATIVE"
    return decision

# Choose a sample based on the imbalanced distribution provided
def randomSample(originalDist):
    sum = np.sum(list(originalDist.values()))
    limit = originalDist[0]
    sampleIndex = np.random.randint(sum)
    k1 = 0 if sampleIndex < limit else 1
    j = sampleIndex if sampleIndex < limit else sampleIndex - limit    

    return j, k1

#------------------------------------------------------------------------------
class pairgen(keras.utils.Sequence):
    positive = 0.0
    negative = 1.0

    #------------------------------------------------------------------------------
    def __init__(self, mat_t, pos, neg, batch, aug = None, train = False, distribution = None, unbalanced = False, config = None):
        self.mat_t = mat_t
        self.pos = pos
        self.neg = neg
        self.distribution_Pos_Neg = {0: pos / (pos+neg), 1: neg / (pos+neg)}
        self.batch = batch
        self.n_k = mat_t.shape[0] # Number of classes
        self.n_s = mat_t.shape[1] # Size of data samples
        self.shape = mat_t.shape[2:] # Size of image (H x V x C)
        self.aug = aug
        self.train = train
        self.distribution = distribution
        self.unbalanced = unbalanced
        self.count = {"Num_negatives": 0, "Num_samples_total": 0}
        self.config = config

        
    def generatePairPositive(self, j1, k1, x1, x, y, labels_covid, config):
        # For each j1 sample, choose a different j2 sample from the same class
        for j2 in np.random.choice(self.n_s, self.pos, True): 
            if self.n_s == 1: # Exception case where only one shot, must use same image
                j2 = j1 
            else:
                while j1 == j2:
                    j2 = np.random.choice(self.n_s)
            x2 = self.mat_t[k1][j2]
            # x2 = utilLoad.read_set_img(x2, config)

            # if self.aug:
            #     x1['augm'], x2['augm'] = self.aug, self.aug 
                # x1 = apply_data_augmentation(x1, self.aug)
                # x2 = apply_data_augmentation(x2, self.aug)

            if np.random.choice([True, False]):
                x['i1'].append(x1)
                x['i2'].append(x2)
            else:
                x['i1'].append(x2)
                x['i2'].append(x1)

            x['i1_augm'].append(self.aug)
            x['i2_augm'].append(self.aug)
            labels_covid.append([k1,k1])
            y.append(pairgen.positive)

        return x, y, labels_covid

    def generatePairNegative(self, k1, x1, x, y, labels_covid, config):
        # Choice of different class than k1. (Expecting not only 2 classes)
        for k2 in np.random.choice(self.n_k, self.neg, True):
            while k1 == k2:
                k2 = np.random.choice(self.n_k)
            j2 = np.random.choice(self.n_s)
            
            x2 = self.mat_t[k2][j2]
            # x2 = utilLoad.read_set_img(x2, config)

            if np.random.choice([True, False]):
                x['i1'].append(x1)
                x['i2'].append(x2)
                labels_covid.append([k1,k2])
            else:
                x['i1'].append(x2)
                x['i2'].append(x1)
                labels_covid.append([k2,k1])

            # Only positives augmented
            x['i1_augm'].append(0)
            x['i2_augm'].append(0)
            y.append(pairgen.negative)

        return x, y, labels_covid

        
        

    #------------------------------------------------------------------------------
    def __len__(self):
        return math.ceil(self.n_s / self.batch)

    #------------------------------------------------------------------------------
    def __getitem__(self, index):
        # first = index * self.batch
        # last = min(self.n_s, first + self.batch)
        # size = int((last - first) * self.n_k * (self.pos + self.neg))
        # x = {
        #     'i1': np.empty((size,) + self.shape),
        #     'i2': np.empty((size,) + self.shape),
        # }

        y = []
        x = {
            'i1': [],
            'i2': [],
            'i1_augm': [],
            'i2_augm': [],
        }
        y = []                
        labels_covid = []
        proportion = [0,0]

        # No Pos or neg, chosen randomly
        if self.unbalanced:
            # for j1 in range(first, last): # In order
            for __ in range(self.batch): # Random
                j1 = np.random.randint(self.n_s)
                for _ in range(self.n_k): # Same quantity of instances than balanced
                    
                    # Class (k) and instance (j) chosen randomly
                    j1, k1 = randomSample(self.distribution[0])
                    j2, k2 = randomSample(self.distribution[0]) 
                    
                    # Access to the image
                    # x1 = self.mat_t[k1][j1]
                    x1 = self.mat_t[k1][j1]
                    # x1 = utilLoad.read_set_img(x1, self.config)

                    x2 = self.mat_t[k2][j2]
                    # x2 = self.mat_t[k2][j2]
                    # x2 = utilLoad.read_set_img(x2, self.config)
                    
                    # # Only augment positive samples
                    # if k1 == k2 and self.aug:
                    #     x1 = apply_data_augmentation(x1, self.aug)
                    #     x2 = apply_data_augmentation(x2, self.aug)

                    x['i1'].append(x1)
                    x['i2'].append(x2)
                    labels_covid.append([k1,k2])

                    self.count["Num_samples_total"] += 1
                    if k1 == k2:
                        x['i1_augm'].append(self.aug)
                        x['i2_augm'].append(self.aug)

                        y.append(pairgen.positive)
                        proportion[0] += 1
                    else:
                        x['i1_augm'].append(0)
                        x['i2_augm'].append(0)

                        y.append(pairgen.negative)
                        self.count["Num_negatives"] += 1
                        # Debug variable
                        self.count["Prop"] = round(self.count["Num_negatives"] / self.count["Num_samples_total"], 4)    
                        proportion[1] += 1
                    
                    

        else:
            # for j1 in range(first, last): # In order
            for __ in range(self.batch): # Random
                j1 = np.random.randint(self.n_s)
                # For each class
                for k1 in range(self.n_k):
                    # x1 = self.mat_t[k1][j1]
                    # Only augment positives
                    x1 = self.mat_t[k1][j1]
                    # x1 = utilLoad.read_set_img(x1, self.config)

                    # Generate an amount of Positive pairs
                    x, y, labels_covid = self.generatePairPositive(j1, k1, x1, x, y, labels_covid, self.config)
                    proportion[0] += self.pos
                    # Generate an amount of Negative pairs
                    x, y, labels_covid = self.generatePairNegative(k1, x1, x, y, labels_covid, self.config)
                    proportion[1] += self.neg





        # if proportion[1] > 0:
        #     print("PROPOTION", proportion, round(proportion[1]/proportion[0], 4))
        x['i1'], x['i2'], y, labels_covid = np.array(x['i1']), np.array(x['i2']), np.array(y), np.array(labels_covid)
        x['i1_augm'], x['i2_augm'] = np.array(x['i1_augm']), np.array(x['i2_augm'])
        # x['i1'], x['i2'], y, labels_covid = x['i1'], x['i2'], np.array(y), np.array(labels_covid)
        return x, y, labels_covid
