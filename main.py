import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

from biorhythm_counter import BiorhythmCounter
from data_processor import DataProcessor, DataReader
from game import Game
from iris_visualiser import IrisVisualiser
from person import Person


def exercise_1():
    person = Person.from_input()
    print(f"Hello {person._name}! You are {person.day_of_life} days old.")
    print(f"Your biorhythm today is:")
    BiorhythmCounter.get_all_diagnosis(person.day_of_life)

def exercise_2():
    Game().game_round()

def exercise_3():
    df = DataReader("datasets/iris_with_errors.csv").read_data()
    data_processor = DataProcessor(df)
    data_processor.display_empty_cells()
    data_processor.fix_to_big_values()
    data_processor.fix_iris_types()

def count_loose_variance(pca: PCA, n_components: int):
    """
    Function counts the ratio of variance that is lost when n_components are removed
    :param pca:
    :param n_components:
    :return:
    """
    i = len(pca.explained_variance_ratio_)- n_components
    return sum(pca.explained_variance_ratio_[i:])/sum(pca.explained_variance_ratio_) * 100

def exercise_4():
    iris = datasets.load_iris()
    y = pd.Series(iris.target, name="FlowerType")
    pca_iris = PCA(n_components=4).fit(iris.data)
    for n in range(1, 5):
        print(f"{n} components: {count_loose_variance(pca_iris, n)}")


    pca = PCA(n_components=2).fit(iris.data)
    pca_iris = pca.fit_transform(iris.data)


    IrisVisualiser.plot_pca_2D(pca_iris, y)
    pca = PCA(n_components=3).fit(iris.data)
    pca_iris = pca.fit_transform(iris.data)
    IrisVisualiser.plot_pca_3D(pca_iris, y)


def exercise_5():
    iris = sns.load_dataset('iris')
    IrisVisualiser(iris).plot("normal")
    data_processor = DataProcessor(iris)
    data_processor.perform_z_score("sepal_length")
    data_processor.perform_z_score("sepal_width")
    z_scored_iris = data_processor.get_data()
    IrisVisualiser(z_scored_iris).plot("z_score")
    data_processor.set_data(iris)
    data_processor.perform_min_max("sepal_length")
    data_processor.perform_min_max("sepal_width")
    iris_min_max = data_processor.get_data()
    IrisVisualiser(iris_min_max).plot("min_max")


if __name__ == "__main__":
    #exercise_1()
    #exercise_2()
    #exercise_3()
    exercise_4()
    #exercise_5()