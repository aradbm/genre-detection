from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from classifier_interface import ClassifierInterface


class KNNClassifier(ClassifierInterface):
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def __str__(self):
        return f"KNN Classifier with {self.model.n_neighbors} neighbors"


if __name__ == '__main__':
    k_values = [3, 5, 7, 9, 13, 17]
    file_paths = ['features/chroma/features_29032024_1938.csv',
                  'features/mfcc/features_29032024_1930.csv',
                  'features/combined_features.csv']
    n_runs = 20

    results = {}

    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        X, y = KNNClassifier.load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for k in k_values:
            accuracies = []
            for seed in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=seed)
                knn = KNNClassifier(n_neighbors=k)
                knn.train(X_train, y_train)
                accuracy = knn.evaluate(X_test, y_test)
                accuracies.append(accuracy)
            average_accuracy = sum(accuracies) / n_runs
            # print(f"Average Accuracy for k={k}: {average_accuracy:.2f}")
            results[(file_path, k)] = average_accuracy

    # If you want to print the overall results in the end
    print()
    print("Overall Results:")
    for file_path in file_paths:
        print(f"File: {file_path}")
        for k in k_values:
            print(f"k={k}: {results[(file_path, k)]:.2f}")
