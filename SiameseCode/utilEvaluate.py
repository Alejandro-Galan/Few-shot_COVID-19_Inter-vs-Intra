import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# AVERAGE = "weighted"
AVERAGE = "macro"

#------------------------------------------------------------------------------
def hist(e_tr, y_tr, e_te, y_te):
    n_k = len(np.unique(y_tr))

    def preddiction(i_te, e_tr):
        def distance(i_tr, i_te):
            return np.linalg.norm(i_tr - i_te)
        return np.argmin(
            np.average(
                np.reshape(
                    np.apply_along_axis(distance, 1, e_tr, i_te),
                    (n_k, -1)
                ),
                1
            )
        )
        # distancias = np.reshape(np.apply_along_axis(distance, 1, e_tr, i_te),
        #                         (len(np.unique(y_te)), -1))
        # medias = np.average(distancias, 1)
        # indice = np.argmin(medias)

    p_te = np.apply_along_axis(preddiction, 1, e_te, e_tr)

    acc    = (y_te == p_te).mean()
    # Assumption of just 2 classes
    TP, TN = ( p_te[y_te == 1] == y_te[y_te == 1]).sum(), ( p_te[y_te == 0] == y_te[y_te == 0]).sum()
    FP, FN = ( p_te[y_te == 1] != y_te[y_te == 1] ).sum(), ( p_te[y_te == 0] != y_te[y_te == 0]).sum()

    # print("TP TN FP FN")
    # print(TP, TN, FP, FN)

    #precision = TP / (TP + FP)
    precision = precision_score(y_te, p_te, average=AVERAGE)    
    #recall    = TP / (TP + FN)
    recall = recall_score(y_te, p_te, average=AVERAGE)    
    #f1        = 2 * precision * recall / (precision + recall)
    f1 = f1_score(y_te, p_te, average=AVERAGE)    
    

    # print('Accuracy on test set (hist):  %0.2f%%' % (acc * 100))
    # print('Precision on test set (hist): %0.2f%%' % (precision * 100))
    # print('Recall on test set (hist):    %0.2f%%' % (recall * 100))
    # print('F1 on test set (hist):        %0.2f%%' % (f1 * 100))

    return p_te, [acc, precision, recall, f1]


#------------------------------------------------------------------------------
def svr(e_tr, y_tr, e_te, y_te, kernel='rbf', return_pred = False):
    #clf = SVR(kernel=kernel, gamma='auto')
    clf = SVC(C=100, kernel=kernel, gamma='auto')
    clf.fit(e_tr, y_tr)
    acc = clf.score(e_te, y_te)
    
    
    # Assumption of just 2 classes
    predicted_y = clf.predict(e_te)
    TN, FP, FN, TP = confusion_matrix(y_te, predicted_y).ravel()

    # print("TP TN FP FN")
    # print(TP, TN, FP, FN)

    #precision = TP / (TP + FP)
    precision = precision_score(y_te, predicted_y, average=AVERAGE)    
    #recall    = TP / (TP + FN)
    recall = recall_score(y_te, predicted_y, average=AVERAGE)    
    #f1        = 2 * precision * recall / (precision + recall)
    f1 = f1_score(y_te, predicted_y, average=AVERAGE)    
    

    # print('Accuracy on test set (svr):  %0.2f%%' % (acc * 100))
    # print('Precision on test set (svr): %0.2f%%' % (precision * 100))
    # print('Recall on test set (svr):    %0.2f%%' % (recall * 100))
    # print('F1 on test set (svr):        %0.2f%%' % (f1 * 100))
        
    return clf.predict(e_te), [acc, precision, recall, f1]



#------------------------------------------------------------------------------
def rf(e_tr, y_tr, e_te, y_te, n_estimators=100, return_pred = False):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(e_tr, y_tr)
    acc = clf.score(e_te, y_te)

    # Assumption of just 2 classes
    predicted_y = clf.predict(e_te)
    TN, FP, FN, TP = confusion_matrix(y_te, predicted_y).ravel()

    # print("TP TN FP FN")
    # print(TP, TN, FP, FN)

    #precision = TP / (TP + FP)
    precision = precision_score(y_te, predicted_y, average=AVERAGE)    
    #recall    = TP / (TP + FN)
    recall = recall_score(y_te, predicted_y, average=AVERAGE)    
    #f1        = 2 * precision * recall / (precision + recall)
    f1 = f1_score(y_te, predicted_y, average=AVERAGE)    
    

    # print('Accuracy on test set (rf):  %0.2f%%' % (acc * 100))
    # print('Precision on test set (rf): %0.2f%%' % (precision * 100))
    # print('Recall on test set (rf):    %0.2f%%' % (recall * 100))
    # print('F1 on test set (rf):        %0.2f%%' % (f1 * 100))


    return clf.predict(e_te), [acc, precision, recall, f1]


#------------------------------------------------------------------------------
def knn(e_tr, y_tr, e_te, y_te, n_neighbors=1, return_pred = False):
    if len(y_tr) < n_neighbors or len(e_te) < n_neighbors:
        # print("Num_samples changed from", n_neighbors, "to", min(len(y_tr), len(e_te)) )
        n_neighbors = min(len(e_te), len(y_tr))

    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(e_tr, y_tr)
    acc = clf.score(e_te, y_te)

   # Assumption of just 2 classes
    predicted_y = clf.predict(e_te)
    TN, FP, FN, TP = confusion_matrix(y_te, predicted_y).ravel()
    
    # print("TP TN FP FN")
    # print(TP, TN, FP, FN)

    #precision = TP / (TP + FP)
    precision = precision_score(y_te, predicted_y, average=AVERAGE)    
    #recall    = TP / (TP + FN)
    recall = recall_score(y_te, predicted_y, average=AVERAGE)    
    #f1        = 2 * precision * recall / (precision + recall)
    f1 = f1_score(y_te, predicted_y, average=AVERAGE)    
    

    # print('Accuracy on test set (knn):  %0.2f%%' % (acc * 100))
    # print('Precision on test set (knn): %0.2f%%' % (precision * 100))
    # print('Recall on test set (knn):    %0.2f%%' % (recall * 100))
    # print('F1 on test set (knn):        %0.2f%%' % (f1 * 100))

    return predicted_y, [acc, precision, recall, f1]

