from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from pandas import isnull



class DataReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read_data(self) -> pd.DataFrame:
        missing_values = ["n/a", "nan", "-"]
        return pd.read_csv(filepath_or_buffer="datasets/iris_with_errors.csv", na_values=missing_values)

class DataProcessor:

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def display_empty_cells(self):
        print(self.data.isnull().sum())

    def fix_to_big_values(self):
        for column in self.data.select_dtypes(include=[np.number]).columns:
            median = self.data[column].median()
            self.data.loc[self.data[column] > 15, column] = median
            self.data.loc[self.data[column] < 0, column] = median
            self.data[column].fillna(median, inplace=True)

    def _find_similar_species(self, species: str) -> str:
        species_list = ['Setosa', 'Versicolor', 'Virginica']
        for s in species_list:
            if SequenceMatcher(None, species, s).ratio() > 0.5:
                return s
        return species
    def fix_iris_types(self):
        species_list = ['Setosa', 'Versicolor', 'Virginica']
        n_wrong_species = len(self.data) - len(self.data[self.data['variety'].isin(species_list)])
        print("Wrong species count: ", n_wrong_species)
        for idx, row in self.data.iterrows():
            if row['variety'] not in species_list:
                self.data.at[idx, 'variety'] = self._find_similar_species(row['variety'])
        n_wrong_species = len(self.data) - len(self.data[self.data['variety'].isin(species_list)])
        print("Wrong species count: ", n_wrong_species)

    def perform_z_score(self, column:str):
        self.data[column] = (self.data[column] - self.data[column].mean()) / self.data[column].std()

    def perform_min_max(self, column:str):
        self.data[column] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())

    def get_data(self) -> pd.DataFrame:
        return self.data

    def set_data(self, data: pd.DataFrame)->None:
        self.data = data