import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, hidden_size)
        self.l4 = torch.nn.Linear(hidden_size, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.softmax(out)
        return out

    def __str__(self):
        return 'Neural Network'


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['file', 'genre'], axis=1).values.astype('float32')
    y = LabelEncoder().fit_transform(data['genre'].values)
    return torch.tensor(X), torch.tensor(y, dtype=torch.long)


def train(model, X_train, y_train, num_epochs, learning_rate, batch_size=64):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
        print(f'Accuracy: {100 * correct / total}%')


if __name__ == '__main__':
    file_path = 'features/chroma/features_29032024_1938.csv'
    # file_path = 'features/mfcc/features_29032024_1930.csv'
    X, y = load_data(file_path)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    input_size = X.shape[1]
    hidden_size = 600
    num_classes = len(torch.unique(y))

    model = NeuralNetwork(input_size, hidden_size, num_classes)
    train(model, x_train, y_train, num_epochs=5000, learning_rate=0.001)
    evaluate(model, x_test, y_test)
    print(model)
