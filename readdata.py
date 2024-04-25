import numpy as np
import pandas as pd


def readdata(PATH):
    if PATH == "breast":
        DATA_PATH = "data/breast-cancer-wisconsin.data"
        columnNames = ['Sample code number',
                       'Clump Thickness',
                       'Uniformity of Cell Size',
                       'Uniformity of Cell Shape',
                       'Marginal Adhesion',
                       'Single Epithelial Cell Size',
                       'Bare Nuclei',
                       'Bland Chromatin',
                       'Normal Nucleoli',
                       'Mitoses',
                       'Class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        data = data.replace(to_replace="?", value=np.nan)
        data = data.dropna(how='any')
        feature = np.array(data[columnNames[1:10]])  # features
        target = np.array(data[columnNames[10]]/2 - 1)  # labels
    elif PATH == "blood-transfusion":
        DATA_PATH = "data/blood-transfusion.data"
        data = pd.read_csv(DATA_PATH)
        data = np.array(data)
        feature = data[:, 0:4]  # features
        target = data[:, 4]  # labels
    elif PATH == "glass":
        DATA_PATH = "data/glass.data"
        columnNames = ['ID number',
                       'refractive index',
                       'Sodium',
                       'Magnesium',
                       'Aluminum',
                       'Silicon',
                       'Potassium',
                       'Calcium',
                       'Barium',
                       'Iron',
                       'Class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        feature = np.array(data[columnNames[1:10]])  # features
        target = np.array(data[columnNames[10]] - 1)  # labels
        target = np.where(target < 3, target, target - 1)
    elif PATH == "heart":
        DATA_PATH = "data/heart.csv"
        data = pd.read_csv(DATA_PATH)
        indf = data['thal'].str.startswith('f')
        indn = data['thal'].str.startswith('n')
        indr = data['thal'].str.startswith('r')
        data.loc[indn, 'thal'] = 3
        data.loc[indf, 'thal'] = 6
        data.loc[indr, 'thal'] = 7
        data = np.array(data)
        feature = data[:, 0:13]  # features
        target = data[:, 13]  # labels
    elif PATH == "ionosphere":
        DATA_PATH = "data/ionosphere.data"
        c1 = list(range(34))
        c2 = ['type']
        columnNames = c1 + c2
        data = pd.read_csv(DATA_PATH, names=columnNames)
        indg = data['type'].str.startswith('g')
        indd = data['type'].str.startswith('b')
        data.loc[indg, 'type'] = 0
        data.loc[indd, 'type'] = 1
        feature = np.array(data[columnNames[0:34]])  # features
        target = np.array(data[columnNames[34]])  # labels
    elif PATH == "parkinsons":
        DATA_PATH = "data/parkinsons.data"
        data = pd.read_csv(DATA_PATH)
        data = np.array(data)
        feature = np.hstack((data[:, 1:17], data[:, 18:24]))  # features
        target = data[:, 17]  # labels
    elif PATH == "wine":
        DATA_PATH = "data/wine.data"
        columnNames = ['Class',
                       'Alcohol',
                       'Malic acid',
                       'Ash',
                       'Alcalinity of ash',
                       'Magnesium',
                       'Total phenols',
                       'Flavanoids',
                       'Nonflavanoid phenols',
                       'Proanthocyanins',
                       'Color intensity',
                       'Hue',
                       'OD280/OD315 of diluted wines',
                       'Proline']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        feature = np.array(data[columnNames[1:14]])  # features
        target = np.array(data[columnNames[0]] - 1)  # labels
    elif PATH == "car":
        DATA_PATH = "data/car.data"
        columnNames = ['buying',
                       'maint',
                       'doors',
                       'persons',
                       'lug_boot',
                       'safety',
                       'Class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        # 数字化处理
        ind11 = data["buying"].str.startswith('v')
        ind12 = data["buying"].str.startswith('h')
        ind13 = data["buying"].str.startswith('m')
        ind14 = data["buying"].str.startswith('l')
        data.loc[ind11, 'buying'] = 4
        data.loc[ind12, 'buying'] = 3
        data.loc[ind13, 'buying'] = 2
        data.loc[ind14, 'buying'] = 1
        ind21 = data["maint"].str.startswith('v')
        ind22 = data["maint"].str.startswith('h')
        ind23 = data["maint"].str.startswith('m')
        ind24 = data["maint"].str.startswith('l')
        data.loc[ind21, 'maint'] = 4
        data.loc[ind22, 'maint'] = 3
        data.loc[ind23, 'maint'] = 2
        data.loc[ind24, 'maint'] = 1
        ind31 = data["doors"].str.startswith('5')
        ind41 = data["persons"].str.startswith('m')
        data.loc[ind31, 'doors'] = 5
        data.loc[ind41, 'persons'] = 6
        ind51 = data["lug_boot"].str.startswith('s')
        ind52 = data["lug_boot"].str.startswith('m')
        ind53 = data["lug_boot"].str.startswith('b')
        data.loc[ind51, 'lug_boot'] = 1
        data.loc[ind52, 'lug_boot'] = 2
        data.loc[ind53, 'lug_boot'] = 3
        ind61 = data["safety"].str.startswith('l')
        ind62 = data["safety"].str.startswith('m')
        ind63 = data["safety"].str.startswith('h')
        data.loc[ind61, 'safety'] = 1
        data.loc[ind62, 'safety'] = 2
        data.loc[ind63, 'safety'] = 3
        ind71 = data["Class"].str.startswith('u')
        ind72 = data["Class"].str.startswith('a')
        ind73 = data["Class"].str.startswith('g')
        ind74 = data["Class"].str.startswith('v')
        data.loc[ind71, 'Class'] = 0
        data.loc[ind72, 'Class'] = 1
        data.loc[ind73, 'Class'] = 2
        data.loc[ind74, 'Class'] = 3
        feature = np.array(data[columnNames[0:6]])  # features
        target = np.array(data[columnNames[6]])  # labels
    elif PATH == "iris":
        DATA_PATH = "data/iris.data"
        columnNames = ['sepal length',
                       'sepal width',
                       'petal length',
                       'petal width',
                       'class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        ind1 = data["class"].str.startswith('Iris-setosa')
        ind2 = data["class"].str.startswith('Iris-versicolor')
        ind3 = data["class"].str.startswith('Iris-virginica')
        data.loc[ind1, 'class'] = 0
        data.loc[ind2, 'class'] = 1
        data.loc[ind3, 'class'] = 2
        feature = np.array(data[columnNames[0:4]])  # features
        target = np.array(data[columnNames[4]])  # labels
    elif PATH == "seeds":
        data = np.loadtxt('data/seeds_dataset.txt')
        feature = data[:, 0:7]
        target = data[:, 7] - 1
    elif PATH == "raisin":
        data = pd.read_excel('data/Raisin_Dataset.xlsx')
        ind1 = data["Class"].str.startswith('K')
        ind2 = data["Class"].str.startswith('B')
        data.loc[ind1, 'Class'] = 0
        data.loc[ind2, 'Class'] = 1
        feature = np.array(data.values[:, 0:7])  # features
        target = np.array(data.values[:, 7])  # labels
    elif PATH == "caesarian":
        DATA_PATH = "data/caesarian.data"
        columnNames = ['Age',
                       'Delivery number',
                       'Delivery time',
                       'Blood of Pressure',
                       'Heart Problem',
                       'class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        feature = np.array(data[columnNames[0:5]])  # features
        target = np.array(data[columnNames[5]])  # labels
    elif PATH == "ecoli":
        DATA_PATH = "data/ecoli.data"
        columnNames = ['mcg',
                       'gvh',
                       'lip',
                       'chg',
                       'aac',
                       'alm1',
                       'alm2',
                       'Class']
        data = pd.read_csv(DATA_PATH, names=columnNames)
        ind1 = data["Class"].str.startswith('cp')
        ind2 = data["Class"].str.startswith('im')
        ind3 = data["Class"].str.startswith('pp')
        ind4 = data["Class"].str.startswith('U')
        ind5 = data["Class"].str.startswith('om')
        data.loc[ind1, 'Class'] = 0
        data.loc[ind2, 'Class'] = 1
        data.loc[ind3, 'Class'] = 2
        data.loc[ind4, 'Class'] = 3
        data.loc[ind5, 'Class'] = 4
        feature = np.array(data[columnNames[0:7]])  # features
        target = np.array(data[columnNames[7]])  # labels

    target = np.array(target, dtype=int)
    return feature, target

