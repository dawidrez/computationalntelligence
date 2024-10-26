from enum import Enum

from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class IrisTypes(str, Enum):
    Setosa = "Setosa"
    Versicolor = "Versicolor"
    Virginica = "Virginica"

def classify_iris(sl: float, sw: float, pl:float,pw:float) -> str:
    """
    Function classifies iris based on its features
    :param sl:
    :param sw:
    :param pl:
    :param pw:
    :return:
    """
    if pw <1:
         return IrisTypes.Setosa.value
    elif pl<5:
        return IrisTypes.Versicolor.value
    return IrisTypes.Virginica.value

def get_train_test_split() -> (pd.DataFrame, pd.DataFrame):
    """
    Function reads iris dataset and splits it into train and test set
    :return:
    """
    df = pd.read_csv("datasets/iris.csv")
    return train_test_split(df.values, train_size=0.7, random_state=299387)

def exercise_1():
    (train_set, test_set) = get_train_test_split()
    good_predictions = 0
    len = test_set.shape[0]
    for i in range(len):
        if classify_iris(test_set[i][0], test_set[i][1],test_set[i][2], test_set[i][3]) == test_set[i][4]:
            good_predictions +=1

    print(good_predictions/len*100)

def exercise_2():
    (train_set, test_set) = get_train_test_split()
    c_tree = tree.DecisionTreeClassifier()
    c_tree.fit(train_set[:, 0:4], train_set[:, 4])
    plt.figure(figsize=(12, 8))
    tree.plot_tree(c_tree, filled=True, feature_names=train_set.columns[0:4], class_names=train_set['variety'].unique())
    plt.show()
    print(c_tree.score(test_set[:, 0:4], test_set[:, 4])*100)
    predictions = c_tree.predict(test_set[:, 0:4])
    print(confusion_matrix(test_set[:, 4], predictions))


def knn_classify(k:int, train_set, test_set ):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_set[:, 0:4], train_set[:, 4])
    y_pred = knn.predict(test_set[:, 0:4])
    print(f"{k} - {accuracy_score(test_set[:, 4], y_pred)*100}")
    print(confusion_matrix(test_set[:, 4], y_pred))



def exercise_3():
    (train_set, test_set) = get_train_test_split()
    # compare different k values
    for k in [3,5,11]:
        knn_classify(k, train_set, test_set)

    # compare naive bayes
    model = GaussianNB()
    model.fit(train_set[:, 0:4], train_set[:, 4])
    y_pred = model.predict(test_set[:, 0:4])
    print(accuracy_score(test_set[:, 4], y_pred)*100)
    print(confusion_matrix(test_set[:, 4], y_pred))


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()