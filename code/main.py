import os

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model  # 序贯式模型
from tensorflow.keras.layers import Dense, Dropout
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load
from utils import *
import threading
import warnings
warnings.filterwarnings("ignore")

save_lock = threading.Lock()


def DNN(X_train, y_train, iteration_index, IF_TRAIN=True):
    # build the neural net
    seed = 114
    np.random.seed(seed)  # for reproducibility
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    cvloss = []
    cvepoch = []
    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='min',
                                   min_delta=0.001,
                                   patience=25,
                                   verbose=2
                                   )
    batch_size = 720
    nb_epoch = 300
    loss_function = 'categorical_crossentropy'

    lr = 0.0005
    adam = keras.optimizers.Adam(lr=lr)


    nb_classes = 3
    input_dim = X_train.shape[1]
    #input_dim = len(chosen_elements)  # 网络输入维度为微量元素数目
    i = 0
    if IF_TRAIN:
        for train, validation in kfold.split(X_train, y_train):
            model = Sequential()  # 序贯式模型
            model.add(Dense(50, activation='relu', input_shape=(input_dim,)))
            model.add(Dropout(0.1))

            model.add(Dense(100, activation='relu'))
            model.add(Dropout(0.1))

            model.add(Dense(100, activation='relu'))
            model.add(Dropout(0.1))

            model.add(Dense(50, activation='relu'))
            model.add(Dropout(0.1))

            model.add(Dense(25, activation='relu'))
            model.add(Dropout(0.1))
            model.add(Dense(nb_classes, activation='softmax'))

            model.compile(
                loss=loss_function,  # 对数损失
                optimizer=adam,
                metrics=['accuracy']
            )
            h = model.fit(
                X_train[train], keras.utils.to_categorical(y_train[train], nb_classes),
                batch_size=batch_size,
                epochs=nb_epoch,
                verbose=0,  # 日志显示
                validation_data=(X_train[validation], keras.utils.to_categorical(y_train[validation], nb_classes)),
                callbacks=[early_stopping]
            )  # fit将模型训练epochs轮
            plot_acc_loss(
                h,
                len(h.history['loss']),
                fig_path,
                savefig=True,
                figname="DNN_training_" + str(i)
            )

            # test the model
            scores = model.evaluate(X_train[validation], keras.utils.to_categorical(y_train[validation], nb_classes),
                                   verbose=0)  # evaluate函数按batch计算在某些输入数据上模型的误差
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            print("{0}: {1:.2f}".format(model.metrics_names[0], scores[0]))
            cvscores.append(scores[1] * 100)
            cvloss.append(scores[0])
            cvepoch.append(len(h.history['loss']))
            i = i + 1
    else:

        model = keras.models.load_model(model_path + "DNN_best_model.h5")

    return model, np.mean(cvscores)


def my_function():
    return 0


def predict(models, X_predict):
    predictions = pd.DataFrame(data=None)

    for name, model in models.items():
        y_pred = model.predict(X_predict)
        predictions[name + " predict_classes"] = y_pred

    return predictions


def main():
    global  data_path, model_path, result_path, fig_path
    data_path = "../data/"
    model_path = '../model/'
    result_path = "../result/"
    fig_path = "../fig/"
    #file = "Earthchem_train_set_after_66Ma_without_external_sets.xlsx"
    file = "Train_data_set.xlsx"

    # Bulding the dir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # 1. Load data
    print("-----------Loading Dataset----------------")
    data = load_data(data_path + file)
    #data = data[data['TECTONIC SETTING'] != 'INTRAPLATE VOLCANICS'].reset_index(drop=True)


    # 2. Preprocessing dataset
    # 1) Normalize data
    data_x = data.loc[:, elements_upper]

    X = normalize_data(data_x, data_x, trace_element_upper, trace_element_upper)
    data['m_label'] = data['m_type'].map({'DM': 0, 'EM': 1, 'HM': 2})
    y = data['m_label'].values

    ## 2) Choose Features: Feature Importance
    top_features = choose_features(X, y)

    # 3. Train models
    #data_x = data.loc[:, top_features]
    data_x = data.loc[:, elements_upper]
    X = normalize_data(data_x, data_x, trace_element_upper, trace_element_upper, method="CLR")
    y = data['m_label'].values

    print("-----------Training models----------------")

    method = "DNN"
    if method == "DNN":
        best_model = keras.models.load_model(model_path + method + "_best_model.h5")
        #best_model = keras.models.load_model("../model/" + "DNN_3_这是目前最好的结果.h5")
    #elif method == "SVM":
        #best_model = load("../model/" + "svm_model_best.pkl")
    else:
        best_model = load(model_path + method + "_best_model.pkl")
    fig_path = fig_path + method+"/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    # 4. Testing on independent set
    # Depleted mantle basalt

    dataPath = "..//data//independent_test//"

    # 1) Classical basalt testing
    ### Depleted mantle basalt
    DM_basalts1 = pd.read_csv(dataPath + "DM//Mid-Atlantic ridge Earthchem.csv")
    DM_basalts2 = pd.read_excel(dataPath + "DM//Pacific_antarctic ridge Earthchem.xlsx")
    DM_basalts3 = pd.read_excel(dataPath + "DM//East Pacific Rise.xlsx")

    DM_basalts1 = DM_basalts1.rename(columns=dict(zip(major_element, major_element_upper)))
    DM_basalts1 = DM_basalts1.rename(columns=dict(zip(trace_element, trace_element_upper)))

    ### Enriched mantle basalt
    EM_basalts1 = pd.read_csv(dataPath + "EM//DECCAN.csv", encoding='ISO-8859-1')
    EM_basalts2_0 = pd.read_csv(dataPath + "EM//HAWAIIAN_ISLANDS.csv")
    EM_basalts2_1 = pd.read_csv(dataPath + "EM//SAMOAN_ISLANDS.csv", encoding='ISO-8859-1')


    ### Hydrated mantle basalt
    HM_basalts1 = pd.read_csv(dataPath + "HM//MARIANA_ARC_filtered.csv", encoding='ISO-8859-1')
    HM_basalts2 = pd.read_excel(dataPath + "HM//earthchem_Andean_arc_filtered.xlsx", sheet_name="Sheet1")
    all_basalts = [DM_basalts1, DM_basalts3, EM_basalts1, EM_basalts2_0, EM_basalts2_1, HM_basalts1, HM_basalts2]
    all_basalts_names = ['DM: Mid-Atlantic ridge', 'DM: East Pacific Rise', 'EM: Deccan plateau', 'EM: Hawiian islands', 'EM: Samoan islands', 'HM: Mariana arc', 'HM: Andean arc']

    all_basalts_acc = plt_confusion_matrix(data_x, all_basalts, best_model, method, all_basalts_names)

    plot_accuracy(all_basalts_acc, all_basalts_names)


    ## 2) Young mantle source bars
    # Enriched mantle basalt
    EM_250Ma = pd.read_csv(dataPath + "EM//250Ma_300Ma_SIBERIAN_TRAPS.csv", encoding='ISO-8859-1')
    EM_1300Ma = pd.read_excel(dataPath + "EM//1300Ma_Yanliao_LIP.xlsx")

    # Hydrated mantle basalt
    HM_750Ma = pd.read_excel(dataPath + "HM//750Ma_Wadi Ranga Area_Mauricce2012.xlsx")

    EM_1300Ma = EM_1300Ma.rename(columns=dict(zip(major_element, major_element_upper)))
    EM_1300Ma = EM_1300Ma.rename(columns=dict(zip(trace_element, trace_element_upper)))

    HM_750Ma = HM_750Ma.rename(columns=dict(zip(major_element, major_element_upper)))
    HM_750Ma = HM_750Ma.rename(columns=dict(zip(trace_element, trace_element_upper)))

    all_basalts = [EM_250Ma, EM_1300Ma, HM_750Ma]
    all_basalts_names = ['EM: 250Ma', 'EM: 1300Ma', 'HM: 750Ma']
    location_name = ["Siberian Traps LIP-250Ma", "Yanliao LIP fragment-1300Ma", "Wadi Ranga Area-750Ma"]

    plot_count_bar(best_model, data_x, all_basalts, all_basalts_names, method, location_name)

    ## 3) Complex tecto-magmatic settings pies
    Walvis_ridge = pd.read_csv(dataPath + "Complex//100Ma_WALVIS_RIDGE.csv", encoding='ISO-8859-1')
    Pacific_antarctic_ridge = pd.read_excel(dataPath + "Complex//Pacific_antarctic ridge Earthchem.xlsx")

    Chile_ridge = pd.read_excel(dataPath + "Complex//Chile Ridge (new).xlsx")
    Chile_ridge = Chile_ridge[Chile_ridge["SOURCE"] == "EARTHCHEMDB"]
    Chile_ridge.reset_index(inplace=True, drop=True)

    Ontong_Java_LIP = pd.read_excel(dataPath + "Complex//100Ma_Ontong Java LIP.xlsx")
    Iceland = pd.read_csv(dataPath + "Complex//Earthchem_Iceland.csv")
    Palma = pd.read_excel(dataPath + "Complex//La Palma EM_HM.xlsx")

    Palma = Palma.rename(columns=dict(zip(major_element, major_element_upper)))
    Palma = Palma.rename(columns=dict(zip(trace_element, trace_element_upper)))


    all_basalts = [Walvis_ridge, Pacific_antarctic_ridge, Ontong_Java_LIP, Iceland, Palma]
    all_basalts_names = ['DM: 1', "DM: 2", 'EM: 1', "EM: 2", "EM: 3"]
    location_name = ["Walvis Ridge", "Pacific Antarctic Ridge", "Ontong Java LIP", "Iceland", "La Palma"]

    plot_pie(best_model, data_x, all_basalts, all_basalts_names, method, location_name)


if __name__ == '__main__':
    main()