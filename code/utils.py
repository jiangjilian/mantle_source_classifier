from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
from globalVar import *
from sklearn.metrics import accuracy_score, confusion_matrix
import scikits.bootstrap as bootstrp
import matplotlib.ticker as tkr
from sklearn.impute import KNNImputer

def read_data(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")


def load_data(file_path):
    train_data = read_data(file_path)
    grouped_means = train_data.groupby('type')[elements_upper].mean()
    train_data[elements_upper] = train_data[elements_upper].fillna(grouped_means)
    train_data[elements_upper] = train_data[elements_upper].fillna(train_data.groupby('type')[elements_upper].transform('mean'))


    train_data["main_elements"] = train_data[major_element_upper].sum(axis=1)
    train_data[(train_data["main_elements"] < 102) & (train_data["main_elements"] > 98)]

    train_data.reset_index(inplace=True)

    
    train_data.dropna(subset=elements_upper, how="any", inplace=True)
    train_data.drop(train_data[(train_data[elements_upper].values == 0)].index, inplace=True)

    train_data.reset_index(inplace=True, drop=True)

    train_data.to_excel(data_path + file_path.split(".")[0]+" preprocessing.xlsx", index=False)

    """
    print("---------------------------------")
    print("IAB NUMBER:")
    print(len(train_data[train_data["type"] == "IAB"].index))
    print("ICB NUMBER:")
    print(len(train_data[train_data["type"] == "ICB"].index))
    print("CFB NUMBER:")
    print(len(train_data[train_data["type"] == "CFB"].index))
    print("MORB NUMBER:")
    print(len(train_data[train_data["type"] == "MORB"].index))
    print("OIB NUMBER:")
    print(len(train_data[train_data["type"] == "OIB"].index))
    print("DM NUMBER:")
    print(len(train_data[train_data["mantle type"] == "DM"].index))
    print("EM NUMBER:")
    print(len(train_data[train_data["mantle type"] == "EM"].index))
    print("HM NUMBER:")
    print(len(train_data[train_data["mantle type"] == "HM"].index))
    """

    return train_data

def normalize_data(X_train, X_predict, train_trace_elements, test_trace_elements, method="scaler"):
    if method == "scaler":
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_predict_normalized = scaler.transform(X_predict)

    elif method == "CLR":
        X_predict_normalized = clr_std(X_train, X_predict, train_trace_elements, test_trace_elements)

    return X_predict_normalized


def CLR(x):
    percent_x = (x.T * 100 / x.sum(axis=1)).T
    nomalized_x = percent_x
    return nomalized_x


def choose_features(X, y):
    rf = RandomForestClassifier(random_state=50)
    rf.fit(X, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = []
    sorted_importances = []
    for i in indices:
        sorted_features.append(elements_upper[i])
        sorted_importances.append(importances[i])
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # PLot features importance
    new_indices = list(indices)
    new_indices.reverse()
    sorted_features.reverse()
    print(sorted_features)
    plt.figure(figsize=(10, 16))
    plt.title("Feature importances")
    plt.barh(range(len(importances)), importances[new_indices],
             color="#C1C1C1", align="center", height=0.5, ec='k')
    plt.yticks(range(X.shape[1]), sorted_features)
    plt.ylim([-1, X.shape[1]])

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig(fig_path + "Feature importances.jpg")
    #plt.show()
    top_features = []
    total_contribution = 0
    i = 0
    sorted_features.reverse()
    for importance in sorted_importances:
        total_contribution += importance
        if total_contribution <= 0.9:
            top_features.append(sorted_features[i])
        else:
            break
        i = i + 1

    print("Top features contributing to 90%:", top_features)
    return top_features


def clr_std(x_train, x, train_trace_elements, test_trace_elements):
    x_copy = x.copy()
    x_train_copy = x_train.copy()
    x_copy[test_trace_elements] = x_copy[test_trace_elements] / 10000
    x_train_copy[train_trace_elements] = x_train_copy[train_trace_elements] / 10000
    pro_x = CLR(x_copy)
    pro_x_train = CLR(x_train_copy)
    scaler = StandardScaler().fit(pro_x_train)
    X = scaler.transform(pro_x)
    return X


def plot_acc_loss(h, nb_epoch, figPath, savefig=False, figname=None):
    """
    Given DNN model, plot accuracy and loss curve

    :h: a fitted model such as
        h = model.fit(x_train,y_train,...)
    :nb_epoch: the number of epoch
    :savefig: save the plot figure as 'eps' image format, default: False
    :return
    """
    acc, loss, val_acc, val_loss = [x * 100 for x in h.history['accuracy']], h.history['loss'], [x * 100 for x in
                                                                                                 h.history[
                                                                                                     'val_accuracy']], \
                                   h.history['val_loss']
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))

    axs[0].plot(range(nb_epoch), acc, lw=0.5, color='dodgerblue', label='Train')
    axs[0].plot(range(nb_epoch), val_acc, lw=0.5, color='brown', label='Test')
    # axs[0].set_title('Accuracy over ' + str(nb_epoch) + ' Epochs', size=15)
    # plt.legend()
    # plt.grid(linestyle='-.')

    axs[1].plot(range(nb_epoch), loss, lw=0.5, color='dodgerblue', label='Train')
    axs[1].plot(range(nb_epoch), val_loss, lw=0.5, color='brown', label='Test')
    # axs[1].set_title('Loss over ' + str(nb_epoch) + ' Epochs', size=15)
    # plt.legend()

    axs[0].set_ylim(ymax=100)
    axs[1].set_ylim(ymin=0, ymax=1.0)
    # axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Only draw spine between the y-ticks
    # axs[0].spines['left'].set_bounds(-1, 1)
    # axs[1].spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axs[0].yaxis.set_ticks_position('left')
    axs[1].yaxis.set_ticks_position('left')
    axs[0].xaxis.set_ticks_position('bottom')
    axs[1].xaxis.set_ticks_position('bottom')

    axs[0].tick_params(labelsize=13)
    axs[1].tick_params(labelsize=13)

    # plt.grid(linestyle='-.')
    plt.tight_layout()
    path = figPath + "train\\"
    if not os.path.exists(path):
        os.makedirs(path)
    if savefig:
        plt.savefig(path + str(figname) + 'acc_loss.png')
        plt.savefig(path + str(figname) + 'acc_loss.pdf', dpi=300)
        # plt.savefig(path +str(figname)+'acc_loss.svg')
    # plt.savefig(str(nb_epoch)+'_'+str(batch_size)+'_'+str(loss_function)+'_'+str(op_name)+'_'+str(learning_rate)+'.jpg', dpi=300)
    # plt.show()


# Predict and save the results
def predict_and_save(model, data_x, data, output_path, method, name):
    data["acutal_classes"] = name.split(":")[0]
    data["acutal_type"] = data["acutal_classes"].map({"DM": 0, "EM": 1, "HM": 2})

    # Predict using the saved SVM model
    #data[elements_upper] = data[elements_upper].fillna(data[elements_upper].mean())
    imputer = KNNImputer(n_neighbors=5)
    data[elements_upper]=pd.DataFrame(imputer.fit_transform(data[elements_upper]), columns=elements_upper)

    data.dropna(subset=elements_upper, inplace=True)
    data.reset_index(inplace=True, drop=True)
    data = data[(data[elements_upper] != 0).all(axis=1)]
    data.reset_index(drop=True, inplace=True)

    trace_element_upper = [s.upper() + "(PPM)" for s in trace_element]
    trace_element_upper = [elem for elem in trace_element_upper if elem in elements_upper]
    #pred_x = clr_std(data_x, data.loc[:, top_features], trace_element_upper, trace_element_upper)
    pred_x = normalize_data(data_x, data[elements_upper], trace_element_upper, trace_element_upper, method="CLR")

    if method == "DNN":
        pred_y = np.argmax(model.predict(pred_x), axis=-1)
    else:
        pred_y = model.predict(pred_x)
    data[name+' predict_classes'] = pred_y.copy()
    data[name+' predict_classes'].replace({0: "DM", 1: "EM", 2: "HM"}, inplace=True)

    data.to_csv(output_path + name + "_clean.csv")
    acc = accuracy_score(data["acutal_type"], pred_y)

    return acc, pred_y, data["acutal_type"].copy()


def plot_count_bar(model, all_X, dataset, data_name, method, location_name):
    # 设置全局字体和字体大小
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 10})

    fig, axs = plt.subplots(1, len(location_name), figsize=(len(location_name)*3+2, 3))
    bar_width = 0.5
    # axs = axs.flatten()
    all_labels = [0, 1, 2]
    colors = np.array(["#ff7f0e", "#1f77b4", "#2ca02c"])
    for i, (basalt, name) in enumerate(zip(dataset, data_name)):
        acc, y_pred, y_true = predict_and_save(model, all_X, basalt, result_path, method, name)
        unique_labels, counts = np.unique(y_pred, return_counts=True)
        pred_matrix = pd.DataFrame(np.unique(y_pred, return_counts=True)).T.sort_values(by=0)
        # unique_labels_replaced = np.where(unique_labels == 0, "DM",
        #                                  np.where(unique_labels == 1, "EM", "HM"))
        # Ensure all mantle types are represented in the counts
        for label in all_labels:
            if label not in unique_labels:
                unique_labels = np.append(unique_labels, label)
                counts = np.append(counts, 0)

        # Sort the labels and counts to maintain consistency
        sorted_indices = np.argsort(unique_labels)
        unique_labels = unique_labels[sorted_indices]
        counts = counts[sorted_indices]

        unique_labels_replaced = np.where(unique_labels == 0, "DM",
                                          np.where(unique_labels == 1, "EM", "HM"))
        # Ensure EM is always on the left, followed by DM and HM
        sorted_labels = ["EM", "DM", "HM"]
        sorted_counts = [counts[np.where(unique_labels_replaced == label)[0][0]] for label in sorted_labels]
        axs[i].bar(sorted_labels, sorted_counts, color=colors[list(map(int, unique_labels.tolist()))], width=bar_width)
        axs[i].set_title(location_name[i])
        axs[i].set_xlabel('Mantle Type')
        axs[i].set_ylabel('Count')
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].tick_params(axis='both', length=5, width=1)
        axs[i].tick_params(axis='both', direction='in')
        axs[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if i == 0:
            axs[i].set_ylim(0, 4000)
        elif i == 1:
            axs[i].set_ylim(0, 100)
        else:
            axs[i].set_ylim(0, 12)

    plt.tight_layout()
    plt.savefig(fig_path + "Figure_3_Pure_Basalts_temporal_effectiveness_pie.jpg")
    plt.savefig(fig_path + "Figure_3_Pure_Basalts_temporal_effectiveness_pie.pdf")
    plt.close()


def plot_pie(model, all_X, dataset, data_name, method, location_name):
    fig, axs = plt.subplots(1, len(location_name), figsize=((len(location_name)-2)*10+2, 10))
    axs = axs.flatten()

    all_labels = [0, 1, 2]
    colors = np.array(["#1f77b4", "#ff7f0e", "#2ca02c"])
    for i, (basalt, name) in enumerate(zip(dataset, data_name)):
        acc, y_pred, y_true = predict_and_save(model, all_X, basalt, result_path, method, name)
        unique_labels, counts = np.unique(y_pred, return_counts=True)
        pred_matrix = pd.DataFrame(np.unique(y_pred, return_counts=True)).T.sort_values(by=0)
        unique_labels_replaced = np.where(unique_labels == 0, "DM",
                                          np.where(unique_labels == 1, "EM", "HM"))
        # axs[i].pie(counts, autopct='%1.f%%', colors=colors[list(map(int, unique_labels.tolist())),])
        axs[i].pie(counts, autopct='%1.f%%', colors=colors[list(map(int, unique_labels.tolist()))], textprops={'fontsize': 24}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        # axs[i].pie(counts, labels=unique_labels_replaced, autopct='%1.f%%', colors=colors[list(map(int, unique_labels.tolist()))], textprops={'fontsize': 24})
        # axs[i].pie(counts, labels=unique_labels, autopct='%1.f%%')
        axs[i].set_title(location_name[i], fontsize=32)
    # 创建图例
    from matplotlib.patches import Patch
    legend_labels = ['Depleted Mantle', 'Enriched Mantle', 'Hydrated Mantle']
    patches = [Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
    #plt.legend(handles=patches, loc='lower center', fontsize=18, ncol=3)
    #plt.legend(handles=patches, loc='lower center', fontsize=18)

    plt.tight_layout()
    plt.savefig(fig_path + "Figure_3_Complex_Basalts_temporal_effectiveness_pie.jpg")
    plt.savefig(fig_path + "Figure_3_Complex_Basalts_temporal_effectiveness_pie.pdf")
    plt.close()


def plt_confusion_matrix(all_X, dataset, model, method, data_name):
    class_names = ["DM", "EM", "HM"]
    # count = 0
    # 设置全局字体和字体大小
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams.update({'font.size': 10})
    all_basalts_acc = []
    all_basalts_pred_y = []
    all_basalts_y = []
    for basalt, name in zip(dataset, data_name):
        acc, y_pred, y_true = predict_and_save(model, all_X, basalt, result_path, method, name)
        all_basalts_acc.append(acc)
        all_basalts_pred_y.append(y_pred)
        all_basalts_y.append(y_true)
        C = confusion_matrix(y_true, y_pred)

        # Check if the third class is missing from the confusion matrix
        unique_classes = np.unique(y_pred)
        if len(unique_classes) < len(class_names):
            missing_classes = [i for i in range(len(class_names)) if i not in unique_classes]
            # Create a new confusion matrix with zeros for missing classes
            new_shape = (len(class_names), len(class_names))
            C_new = np.zeros(new_shape, dtype=int)
            # Fill in the existing values
            for i in range(len(unique_classes)):
                for j in range(len(unique_classes)):
                    C_new[int(unique_classes[i]), int(unique_classes[j])] = C[i, j]
            C = C_new

        C = C[~np.all(C == 0, axis=1)]

        # 绘制混淆矩阵图
        # count += 1
        # plt.subplot(len(all_basalts_names), 1, count)
        plt.matshow(C, cmap=plt.cm.Greens)

        # axs[i, 0].matshow(C, cmap=plt.cm.Greens)
        # plt.colorbar()
        for i in range(len(C)):
            for j in range(len(C[0])):
                plt.annotate(C[i, j], xy=(j, i), horizontalalignment='center', verticalalignment='center', fontsize=12)
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        plt.yticks(ticks=[0], labels=[name.split(":")[0]])
        plt.xticks(ticks=np.arange(len(class_names)), labels=class_names)
        print(fig_path + "confusion matrix/" + name + "_confusion_matrix.jpg")
        if not os.path.exists(fig_path + "confusion matrix/"):
            os.makedirs(fig_path + "confusion matrix/")
        plt.savefig(fig_path + "confusion matrix/" + name.split(":")[1] + "_confusion_matrix.jpg")
        plt.savefig(fig_path + "confusion matrix/" + name.split(":")[1] + "_confusion_matrix.pdf")
    return all_basalts_acc


def plot_accuracy(all_basalts_acc, data_name):
    plt.figure(figsize=(15, 8))
    i = 0
    # 获取 jet 颜色映射
    cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('jet')
    # 计算颜色索引
    colors = [cmap(i / len(all_basalts_acc)) for i in range(len(all_basalts_acc))]

    # 计算all_acc由大到小的索引
    sorted_indices = np.argsort(all_basalts_acc)

    # 创建一个新的颜色列表，按照 sorted_indices 的顺序排列
    sorted_colors = [None] * len(colors)  # 初始化一个空列表

    for new_index, original_index in enumerate(sorted_indices):
        sorted_colors[original_index] = colors[new_index]  # 将原始颜色按新顺序填入

    # 设置点的大小，可以根据 acc 的值进行调整
    # Normalize all_acc
    size = (all_basalts_acc - min(all_basalts_acc)) / (max(all_basalts_acc) - min(all_basalts_acc)) + 0.01
    point_sizes = [600 * i for i in size]  # 这里的 100 是一个缩放因子，可以根据需要调整

    # 绘制散点图
    fig = plt.figure(figsize=(4, 8))
    for acc, label, color in zip(all_basalts_acc, data_name, sorted_colors):
        #plt.scatter(i, acc, label=label, color=color, alpha=1, s=point_sizes[i])
        plt.scatter(acc, i, label=label, color=color, alpha=1, s=380)
        i += 1
    plt.xlabel('Accuracy')
    #plt.ylabel('File Name')
    plt.ylim(-0.3, )
    #plt.xlim(0.4, )
    plt.tick_params(axis='both', length=5, width=1)
    plt.tick_params(axis='both', direction='in')
    #plt.title('Accuracy of Predictions for Each Basalt Type')
    #plt.legend()
    plt.gca().set_yticklabels([])
    plt.xlim(0.5, 1.0)
    plt.savefig(fig_path + 'Figure_2_Accuracy_of_Predictions_for_Each_Basalt_Type2.png')
    plt.savefig(fig_path + 'Figure_2_Accuracy_of_Predictions_for_Each_Basalt_Type2.pdf', dpi=300)
    plt.tight_layout()
    plt.show()


def bootstrap(data, age, type, col, bin_width, MIN_AGE, MAX_AGE, step):

    low_AGE = MAX_AGE - bin_width
    high_AGE = MAX_AGE

    i = 0
    result = pd.DataFrame(data=None)
    while low_AGE >= MIN_AGE:
        data_bin = data[(age < high_AGE) & (age >= low_AGE)]
        sample_num = len(data_bin)

        # Remove outliers
        OutlierL = data_bin.quantile(0.05)
        OutlierH = data_bin.quantile(0.95)
        data_bin[data_bin[(data_bin > OutlierH) | (data_bin < OutlierL)].index] = np.nan
        data_bin = data_bin[data_bin[~np.isnan(data_bin)].index]
        sample_num = len(data_bin)
        # print(nA[j])
        if sample_num >= 4:  # less than 4 samples will not be calculated.
            CIs = bootstrp.ci(data=data_bin, statfunction=np.mean, n_samples=10000)
            result.loc[i, str(type) + " " + str(col) + " CI1"] = CIs[0]
            result.loc[i, str(type) + " " + str(col) + " CI2"] = CIs[1]
            result.loc[i, str(type) + " " + str(col) + " std"] = (CIs[1] - CIs[0]) / 2  # standard error

        else:
            result.loc[i, str(type) + " " + str(col) + " std"] = np.nan  # standard error

        result.loc[i, "AGE_MEDIAN"] = (low_AGE + high_AGE) / 2  # age
        result.loc[i, str(type) + " " + str(col) + " mean"] = np.mean(data_bin)  # mean


        low_AGE = low_AGE - step
        high_AGE = high_AGE - step
        i = i + 1

    result.fillna(result.interpolate(), inplace=True)
    result.fillna(method="pad", inplace=True)
    result.sort_values(by="AGE_MEDIAN", ascending=True, inplace=True)
    result.reset_index(inplace=True, drop=True)

    return result


def plot_fitting_line(data, bin, step, type, col, color, method):
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(direction='in', length=15, width=2, colors='black', grid_color='b', pad=10, which="major")
    ax.tick_params(direction='in', length=10, width=2, colors='black', grid_color='b', pad=10, which="minor")

    ax.xaxis.set_major_locator(tkr.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(500))
    plt.xlim(0, 3000)

    x = data["AGE_MEDIAN"]
    y = data[str(type) + " " + str(col) + " mean"]
    # data[str(index) +" DIFF std"] = data["ICB "+str(index) +" std"] - data["IAB " +str(index) +" std"]
    y_min = data[str(type) + " " + str(col) + " CI1"]
    y_max = data[str(type) + " " + str(col) + " CI2"]

    bar_width = 0.25

    ax.plot(x, y, color='grey')
    line1 = plt.plot(x, y_min, linestyle="--", color='grey')
    line2 = plt.plot(x, y_max, linestyle="--", color='grey')


    X = [[0, 1], [0, 1]]
    xmin = min(x)
    xmax = max(x)
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    xmax = ax.get_xlim()[1]


    cm = plt.cm.get_cmap(color)
    ax.imshow(X, interpolation='bicubic', cmap=cm, extent=(xmin, xmax, ymin, ymax), alpha=1)
    ax.fill_between(x, ymin, y_min, color="white", alpha=1)
    ax.fill_between(x, y_max, ymax, color="white", alpha=1)
    ax.fill_between([2955,3000,3000,2955], [ymin,ymin, ymax, ymax], color="white", alpha=1)
    ax.set_aspect('auto')
    # color="red",


    ax.set_xlabel("AGE(Ma)", fontsize=40, labelpad=10, weight='normal')
    ax.set_ylabel(str(type+ " " + col), fontsize=40, labelpad=10, weight='normal')
    ax.tick_params(labelsize=35, color='black')
    ax.invert_xaxis()

    plt.xticks(fontsize=35, color='black')
    plt.yticks(fontsize=35, color='black')
    # plt.legend(loc="upper right",  fontsize=25)
    plt.tight_layout()

    if not os.path.exists(fig_path + "/" + method + " bootstrap/"):
        os.makedirs(fig_path + "/" + method + " bootstrap/")
    plt.tight_layout()
    fig.savefig(fig_path + "/" + method + " bootstrap/"+ str(type) + "_" + str(col)+"_bin_"+str(bin) + "Ma_step_"+ str(step)+"Ma_boostrap_means.png")
    fig.savefig(fig_path + "/" + method + " bootstrap/"+ str(type) + "_" + str(col)+"_bin_"+str(bin) + "Ma_step_"+ str(step)+"Ma_boostrap_means.pdf")


def plot_DM_EM_HM_element_curve(data, col1, col2, method):
    data["predict_type"] = data[method + " predict_classes"].replace({0: "DM", 1: "EM", 2: "HM"}).copy()
    ## Define the bin size (step width)
    MIN_AGE, MAX_AGE = 0, 4000
    BIN_WIDTH = 500
    STEP = 100
    TYPES = ["DM", "EM", "HM"]
    for TYPE in TYPES:
        new_data = data[data["predict_type"]==TYPE]
        new_data.reset_index(inplace=True, drop=True)
        age = new_data[col1]
        element = new_data[col2]
        result = bootstrap(element, age, TYPE, col2, BIN_WIDTH, MIN_AGE, MAX_AGE, STEP)

        ## Output the bootstrap result
        #print(result)
        if not os.path.exists(result_path + "/" + method + " bootstrap/"):
            os.makedirs(result_path + "/" + method + " bootstrap/")
        result.to_excel(result_path + "/" + method + " bootstrap/" + TYPE +" "+ col2 + " bootstrap.xlsx")

        # Plot a fitting line
        plot_fitting_line(result, BIN_WIDTH, STEP, TYPE, col2, color="Purples", method=method)
