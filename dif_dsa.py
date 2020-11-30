###### different DSA definitions
import numpy as np
import time
import os

from multiprocessing import Pool
from tqdm import tqdm
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score, auc
import matplotlib.pyplot as plt
from utils import *

def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]

def cal_cla_matrix(train_pred):
    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)
    return class_matrix, all_idx

def find_closest_at(at, train_ats,k=1):

    dist = np.linalg.norm(at - train_ats, axis=1)
    # ind=np.argmin(dist)
    ind = np.argsort(dist)  # from min to max
    # mn_dist=min(dist[0:k])
    mn_pt=np.mean(train_ats[ind[0:k]],axis=0)
    mn_dist=np.linalg.norm(at - mn_pt)
    return (mn_dist,mn_pt)

def get_ats( model,dataset, name,layer_names, save_path=None,
             batch_size=128,
             is_classification=True,
             num_classes=10,    num_proc=10,
):
    """Extract activation traces of dataset from model.

    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.

    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output for layer_name in layer_names],
    )

    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")
        pred = model.predict_classes(dataset, batch_size=batch_size, verbose=1)
        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

        print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i] for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    if save_path is not None:
        np.save(save_path[0], ats)
        np.save(save_path[1], pred)

    return ats, pred


##### the original DSA
def cal_dsa0(train_ats, y_train,test_ats, y_test, class_matrix, all_idx):
    dsa = []
    for i, at in enumerate(tqdm(test_ats)):
        label = y_test[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
                    a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
                )
        dsa.append(a_dist / b_dist)

    return dsa

##### the DSA modification 1
def cal_dsa1(train_ats, y_train,test_ats, y_test, class_matrix, all_idx):
    dsa = []
    for i, at in enumerate(tqdm(test_ats)):
        label = y_test[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
                    at, train_ats[list(set(all_idx) - set(class_matrix[label]))]
                )
        dsa.append(a_dist / b_dist)

    return dsa

##### the DSA modification 2
def cal_dsa2(train_ats, y_train,test_ats, y_test, class_matrix):
    dsa = []
    for i, at in enumerate(tqdm(test_ats)):
        label = y_test[i]
        dist = []
        for j in range(10):
            temp = np.linalg.norm(at - np.mean(train_ats[class_matrix[j]], axis=0))
            dist.append(temp)
        a_dist = dist[label]
        b_dist = min(np.delete(dist, label))
        dsa.append(a_dist / b_dist)

    return dsa


##### the DSA modification 3
def cal_dsa3(train_ats, y_train,test_ats, y_test, class_matrix):
    dsa = []
    for i, at in enumerate(tqdm(test_ats)):
        label = y_test[i]
        dist=[]
        for j in range(10):
            temp,_ = find_closest_at(at, train_ats[class_matrix[j]],k=50)
            dist.append(temp)
        a_dist=dist[label]
        b_dist=min(np.delete(dist,label))
        dsa.append(a_dist / b_dist)

    return dsa

#### label of corner case
def pre_cc(model, x_test, y_test,batch_size=128):
    pred = model.predict_classes(x_test, batch_size=batch_size, verbose=1)
    cclab=np.zeros_like(pred)
    cclab[pred!=y_test]=1
    return cclab,pred

### show images
def plt_ccs(data,dsa,num=40,save=False,save_path=None):
    m=np.int(np.ceil(num/8))
    ind=np.argsort(dsa)
    IMG=np.zeros([28*m,28*8])
    for i in range (num):
        row,col=i//8,i%8
        IMG[row*28:(row+1)*28,col*28:(col+1)*28]=data[ind[-i-1]].reshape(28,28)

    fg=plt.figure()
    plt.imshow(IMG,cmap='gray')
    plt.axis('off')
    plt.show()

    if save==True:
        fg.savefig(save_path)

### plot auc-roc
def plt_roc(y_tru,y_pre):
    fpr, tpr, threshold = roc_curve(y_tru, y_pre)
    roc_auc = roc_auc_score(y_tru, y_pre)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

