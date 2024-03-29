from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from classifier_interface import ClassifierInterface


class RFClassifier(ClassifierInterface):
    def __init__(self, n_estimators=100, max_depth=None, criterion='gini', min_samples_split=2, min_samples_leaf=1):
        # Initialize RandomForestClassifier with the given parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy  # Return the accuracy score for further processing

    def __str__(self):
        return f"Random Forest Classifier with {self.model.n_estimators} trees, max_depth={self.model.max_depth}, criterion={self.model.criterion}, min_samples_split={self.model.min_samples_split}, min_samples_leaf={self.model.min_samples_leaf}"


if __name__ == '__main__':
    parameters = [
        {'n_estimators': 100, 'max_depth': 3, 'criterion': 'gini',
         'min_samples_split': 2, 'min_samples_leaf': 1},
        {'n_estimators': 150, 'max_depth': 5, 'criterion': 'gini',
         'min_samples_split': 4, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': None, 'criterion': 'entropy',
         'min_samples_split': 2, 'min_samples_leaf': 1},
    ]
    file_paths = ['features/chroma/features_29032024_1938.csv',
                  'features/mfcc/features_29032024_1930.csv']
    n_runs = 20

    results = {}

    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        X, y = RFClassifier.load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for param in parameters:
            accuracies = []
            for seed in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=seed)
                rf = RFClassifier(n_estimators=param['n_estimators'],
                                  max_depth=param['max_depth'], criterion=param['criterion'],
                                  min_samples_split=param['min_samples_split'], min_samples_leaf=param['min_samples_leaf'])
                rf.train(X_train, y_train)
                accuracy = rf.evaluate(X_test, y_test)
                accuracies.append(accuracy)
            average_accuracy = sum(accuracies) / n_runs
            results[(file_path, str(param))] = average_accuracy

    # Print overall results
    print("\nOverall Results:")
    for file_path in file_paths:
        print(f"File: {file_path}")
        for param in parameters:
            print(f"{param}: {results[(file_path, str(param))]:.2f}")
