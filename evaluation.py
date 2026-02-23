
# this code is released by: 
# L. Yang, X.-Z. Wu, Y. Jiang, and Z.-H. Zhou. 
# Multi-label deep forest. 
#In: Proceedings of the 24th European Conference on Artificial Intelligence (ECAI'20), 
# Santiago de Compostela, Spain, 2020. [code]

import numpy as np
from sklearn import metrics

def do_metric(y_prob, label):

    y_predict = y_prob > 0.5
    ranking_loss = compute_ranking_loss(y_prob, label)
    one_error = compute_one_error(y_prob, label)
    coverage = compute_coverage(y_prob, label)
    hamming_loss = compute_hamming_loss(y_predict, label)
    precision = compute_average_precision(y_prob, label)
    macro_f1 = compute_macro_f1(y_predict, label)
    micro_f1 = compute_micro_f1(y_predict, label)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([precision, 1 - hamming_loss, 1-ranking_loss, auc_me, 1 - one_error, 1 - coverage, auc, macro_f1, micro_f1])

def compute_accuracy(pred_label, label):
    num_samples = len(label)
    acc = sum(label == pred_label) * 1.0 / num_samples
    return acc

def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank

def compute_hamming_loss(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc.mean()

def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1

def compute_micro_f1(pred_label, label):
    up = np.sum(pred_label * label)
    down = np.sum(pred_label) + np.sum(label)
    if np.sum(np.sum(label) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    micro_f1 = 2.0 * up / down
    return micro_f1

def compute_ranking_loss(y_prob, label):
    # y_predict = y_prob > 0.5
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        prob_positive = y_prob[i, label[i, :] > 0.5]
        prob_negative = y_prob[i, label[i, :] < 0.5]
        s = 0
        for j in range(prob_positive.shape[0]):
            for k in range(prob_negative.shape[0]):
                if prob_negative[k] >= prob_positive[j]:
                    s += 1

        label_positive = np.sum(label[i, :] > 0.5)
        label_negative = np.sum(label[i, :] < 0.5)
        if label_negative != 0 and label_positive != 0:
            loss = loss + s * 1.0 / (label_negative * label_positive)

    return loss * 1.0 / num_samples

def compute_one_error(y_prob, label):
    num_samples, num_labels = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples

def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels

def compute_average_precision(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples

def compute_auc(y_prob, label):
    n, m = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(m):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc

def performance(y,f,T):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    n,K = f.shape    
    match = np.zeros(n)
    fn = np.zeros(n)
    fp = np.zeros(n)
    for i in range(n):
        si = f[i,:].argsort()[::-1]        
        words=y[i,:]
        correct_labels=np.where(words>-1)
        correct_labels = (np.array(correct_labels)).reshape(-1)
        si = si[0:T]
        match[i] = 0
        for j in range(len(correct_labels)):
            if np.where(si==correct_labels[j])[0].shape[0]!=0:
                match[i] = match[i]+1
        fn[i] = len(correct_labels)-match[i]
        fp[i] = T-match[i]
    return match,fp,fn
  
def mlr_roc(f, y_test):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL
    K = y_test.shape[1]
    tpr1 = np.zeros(K)
    fpr1 = np.zeros(K)
    
    for i in range(K):
        match,fpp,fnn = performance(y_test,f,i+1);
        tp1=match.sum()
        fn1=fnn.sum()
        fp1=fpp.sum()
        tn1 = K*f.shape[0]-(tp1+fp1+fn1)
        tpr1[i] = tp1/(tp1+fn1)
        fpr1[i] = fp1/(fp1+tn1)
    return tpr1,fpr1

def mlc_auc(rocZ,newY):
#    code is written by Jerry, according to the original code from 
#    from http://mlda.swu.edu.cn/codes.php?name=iMVWL    
#    rocZ: problistic matrix  n*c
#    newY: n*c matrix,elements in {-1,1}
    if newY.min()==0:
        newY = newY*2-1
    
    tpr,fpr = mlr_roc(rocZ,newY)
    area = 0
    m = newY.shape[1]
    for i in range(m-1):
        area = area+(fpr[i+1]-fpr[i])*(tpr[i+1]+tpr[i])*0.5
    value_auc = area/(fpr[m-1]-fpr[0])
    return value_auc

if __name__=='__main__':
    a=np.array([[0.1,0.2,0.1,0.4,0.8,0.9],[0.1,0.2,0.1,0.4,0.8,0.9]])
    b=np.array([[1.,0.,1.,0.,1.,1.],[0.,1.,0.,1.,0.,0.]])
    print(do_metric(a,b))
