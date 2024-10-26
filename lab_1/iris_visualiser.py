import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class IrisVisualiser:
    def __init__(self, iris: pd.DataFrame):
        self.iris = iris

    def plot(self, name:str) -> None:
        sns.set_style("whitegrid")
        sns.FacetGrid(self.iris, hue="species",
                      height=6).map(plt.scatter,
                                    'sepal_length',
                                    'sepal_width').add_legend()
        plt.savefig(f"plots/{name}.png")

    @staticmethod
    def plot_pca_2D(pca_iris: pd.DataFrame, y: pd.Series) -> None:
        plt.scatter(pca_iris[:, 0], pca_iris[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
        plt.title("2D PCA of Iris Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("plots/pca_2D.png")

    @staticmethod
    def plot_pca_3D(pca_iris: pd.DataFrame, y: pd.Series) -> None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_iris[:, 0], pca_iris[:, 1], pca_iris[:, 2], c=y, cmap='viridis', edgecolor='k', s=100)
        plt.title("3D PCA of Iris Dataset")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig("plots/pca_3D.png")