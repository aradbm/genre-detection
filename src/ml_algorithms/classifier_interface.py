from abc import ABC
import pandas as pd


class ClassifierInterface(ABC):
    @staticmethod
    def load_data(file_path):
        data = pd.read_csv(file_path)
        X = data.drop(['file', 'genre'], axis=1)
        y = data['genre']
        return X, y

    def train(self, X_train, y_train):
        pass

    def predict(self, X_test):
        pass

    def evaluate(self, X_test, y_test):
        pass

    def __str__(self):
        pass
