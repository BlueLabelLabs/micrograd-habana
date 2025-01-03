import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from micrograd.utils import get_device_initial

# make up a dataset
from sklearn.datasets import make_moons


def make_dataset():
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y * 2 - 1  # make y be -1 or 1
    return X, y


def initialize_model():
    model = MLP(2, [16, 16, 1], device=get_device_initial())  # 2-layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))
    return model


# loss function
def loss(model, X, y, batch_size=None):
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


def optimize(model, X, y, batch_size=None, num_steps=100):
    for k in range(num_steps):
        # forward
        total_loss, acc = loss(model, X, y, batch_size)

        # backward
        model.zero_grad()
        total_loss.backward()

        # update (sgd)
        learning_rate = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}, accuracy {acc*100:.2f}%")
    return model


def run():
    X, y = make_dataset()
    model = initialize_model()
    optimize(model, X, y, num_steps=3)


if __name__ == "__main__":
    run()
