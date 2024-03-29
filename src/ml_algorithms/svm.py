from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from classifier_interface import ClassifierInterface


class SVMClassifier(ClassifierInterface):
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def __str__(self):
        return f"SVM Classifier with C={self.model.C} and kernel='{self.model.kernel}'"


if __name__ == '__main__':
    configurations = [
        (1.0, 'linear'),
        (1.0, 'rbf'),  # RBF kernel is the default
        (1.0, 'poly'),
        (1.0, 'sigmoid')
    ]
    file_paths = ['features/chroma/features_29032024_1938.csv',
                  'features/mfcc/features_29032024_1930.csv']
    n_runs = 20

    results = {}

    for file_path in file_paths:
        print(f"Evaluating file: {file_path}")
        # Ensure this function is defined or imported correctly
        X, y = SVMClassifier.load_data(file_path)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for C, kernel in configurations:
            accuracies = []
            for seed in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=seed)
                svm_classifier = SVMClassifier(C=C, kernel=kernel)
                svm_classifier.train(X_train, y_train)
                accuracy = svm_classifier.evaluate(X_test, y_test)
                accuracies.append(accuracy)
            average_accuracy = sum(accuracies) / n_runs
            results[(file_path, C, kernel)] = average_accuracy

    print()
    print("Overall Results:")
    for file_path in file_paths:
        print(f"File: {file_path}")
        for C, kernel in configurations:
            print(
                f"C={C}, kernel={kernel}: {results[(file_path, C, kernel)]:.2f}")
