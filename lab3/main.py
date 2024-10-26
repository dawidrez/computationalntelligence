import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def forward_pass(age, weight, height):
    hidden_1 = -0.46122 * age + 0.97314 * weight + -0.39203 * height + 0.80109
    hidden_after_activation = 1 / (1 + np.exp(-hidden_1))
    hidden_2 = 0.78548*age +  2.10584 *weight + -0.57847*height + 0.43529
    hidden_2_after_activation = 1 / (1 + np.exp(-hidden_2))
    return hidden_after_activation * -0.81546 + hidden_2_after_activation * 1.03775 - 0.2368



def exercise_1():
    print(forward_pass(23, 75, 176))
    print(forward_pass(25, 67, 180))

def exercise_2():
    iris = load_iris()
    datasets = train_test_split(iris.data, iris.target,
                                test_size=0.3, random_state=299387)

    train_data, test_data, train_labels, test_labels = datasets
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000)
    mlp.fit(train_data, train_labels)
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels)*100)
    print(confusion_matrix(predictions_test, test_labels))

    mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)
    mlp.fit(train_data, train_labels)
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels) * 100)
    print(confusion_matrix(predictions_test, test_labels))

    mlp = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=2000)
    mlp.fit(train_data, train_labels)
    predictions_test = mlp.predict(test_data)
    print(accuracy_score(predictions_test, test_labels) * 100)
    print(confusion_matrix(predictions_test, test_labels))


def exercise_3():
    df = pd.read_csv("datasets/diabetes.csv")
    (train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=299387)

    mlp = MLPClassifier(hidden_layer_sizes=(6,3), max_iter=500, activation='relu')

    mlp.fit(train_set[:, 0:8], train_set[:, 8])

    predictions = mlp.predict(test_set[:, 0:8])
    print(accuracy_score(predictions, test_set[:, 8])*100)
    print(confusion_matrix(predictions, test_set[:, 8]))

    mlp = MLPClassifier(hidden_layer_sizes=(5,2), max_iter=2000, activation='tanh')

    mlp.fit(train_set[:, 0:8], train_set[:, 8])

    predictions = mlp.predict(test_set[:, 0:8])
    print(accuracy_score(predictions, test_set[:, 8]) * 100)
    print(confusion_matrix(predictions, test_set[:, 8]))
    # negative positive is worse than positive negative


if __name__ == "__main__":
    #exercise_1()
    #exercise_2()
    exercise_3()