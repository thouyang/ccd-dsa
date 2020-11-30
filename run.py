import numpy as np
import time
import argparse

from tqdm import tqdm
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Model

from dif_dsa import *

from utils import *

CLIP_MIN = -0.5
CLIP_MAX = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument(
        "--dsa", "-dsa",
        help="selecting distance-based surprise adequacy (dsa)",type=str,
        default="dsa0",
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="./tmp/"
    )
    parser.add_argument(
        "--batch_size", "-batch_size", help="Batch size", type=int, default=128
    )
    parser.add_argument(
        "--var_threshold", "-var_threshold", help="Variance threshold", type=int,
        default=1e-5,
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    assert args.d in ["mnist","cifar"], "Dataset should be 'mnist'"
    assert args.dsa in ["dsa0","dsa1","dsa2","dsa3"], "Select one of 'dsa' definitions "
    print(args)

    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Load pre-trained model.
        model = load_model("./model/model_mnist.h5")
        model.summary()

        # You can select some layers you want to test.
        layer_names = ["activation_1"]
        # layer_names = ["activation_2"]
        # layer_names = ["activation_3"]

        # Load target set.
        x_target = x_test

    x_train = x_train.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = x_test.astype("float32")
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)
    train_ats, train_pred = get_ats(model, x_train, "train", layer_names)
    test_ats, test_pred = get_ats(model, x_test, "test", layer_names)

    if args.dsa=="dsa0":

        class_matrix, all_idx=cal_cla_matrix(y_train)
        test_dsa = cal_dsa0(train_ats, y_train,test_ats, y_test, class_matrix, all_idx)

        img_path=args.save_path+'corner_cases_dsa0.png'
        plt_ccs(x_test,test_dsa,num=40,save=True,save_path=img_path)
        cc_lab,_=pre_cc(model, x_test, y_test,args.batch_size)
        plt_roc(cc_lab,test_dsa)
        # print(infog("ROC-AUC based on dsa0: " + str(auc * 100)))

    elif args.dsa=="dsa1":
        class_matrix, all_idx=cal_cla_matrix(y_train)
        test_dsa = cal_dsa1(train_ats, y_train,test_ats, y_test, class_matrix, all_idx)

        img_path=args.save_path+'corner_cases_dsa1.png'
        plt_ccs(x_test,test_dsa,num=40,save=True,save_path=img_path)
        cc_lab,_=pre_cc(model, x_test, y_test,args.batch_size)
        plt_roc(cc_lab,test_dsa)

    elif args.dsa=="dsa2":
        class_matrix, all_idx = cal_cla_matrix(y_train)
        test_dsa = cal_dsa2(train_ats, y_train, test_ats, y_test, class_matrix)

        img_path = args.save_path + 'corner_cases_dsa2.png'
        plt_ccs(x_test, test_dsa, num=40,save=True,save_path=img_path)
        cc_lab, _ = pre_cc(model, x_test, y_test, args.batch_size)
        plt_roc(cc_lab, test_dsa)

    elif args.dsa=="dsa3":
        class_matrix, all_idx = cal_cla_matrix(y_train)
        test_dsa = cal_dsa3(train_ats, y_train, test_ats, y_test, class_matrix)

        img_path = args.save_path + 'corner_cases_dsa3.png'
        plt_ccs(x_test, test_dsa, num=40,save=True,save_path=img_path)
        cc_lab, _ = pre_cc(model, x_test, y_test, args.batch_size)
        plt_roc(cc_lab, test_dsa)
        print('complete the processing')

