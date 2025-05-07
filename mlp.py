import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(
            self,
            input_size: int = 784,
            hidden_size: int = 64,
            output_size: int = 10,
            seed: int = 42,
    ) -> None:
        np.random.seed(seed)
        
        # Xavier initialization
        self.W1: np.ndarray = np.random.randn(input_size, hidden_size) * np.sqrt(
            2.0 / (input_size + hidden_size)
        )
        self.b1: np.ndarray = np.zeros((1, hidden_size))
        
        self.W2: np.ndarray = np.random.randn(hidden_size, output_size) * np.sqrt(
            2.0 / (hidden_size + output_size)
        )
        self.b2: np.ndarray = np.zeros((1, output_size))
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))
    
    def _sigmoid_derivative(self, s: np.ndarray) -> np.ndarray:
        return s * (1.0 - s)
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self._sigmoid(self.Z1)
        
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self._softmax(self.Z2)
        
        return self.A2
    
    def compute_loss(self, Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        m = Y_true.shape[0]
        return -float(np.sum(Y_true * np.log(Y_pred + 1e-8)) / m)
    
    def backward(
        self, X: np.ndarray,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        lr: float,
    ) -> None:
        m = X.shape[0]
        dZ2 = (Y_pred - Y_true) / m
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._sigmoid_derivative(self.A1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 10,
        lr: float = 0.1,
        batch_size: int = 64,
    ) -> None:
        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(X.shape[0])
            X_sh, Y_sh = X[idx], Y[idx]
            
            for start in range(0, X.shape[0], batch_size):
                xb = X_sh[start : start + batch_size]
                yb = Y_sh[start : start + batch_size]
                y_pred = self.forward(xb)
                self.backward(xb, yb, y_pred, lr)

            Y_pred_full = self.forward(X)
            loss = self.compute_loss(Y_pred_full, Y)
            acc = float(
                np.mean(np.argmax(Y_pred_full, axis=1) == np.argmax(Y, axis=1))
            )
            print(f"Epoch {epoch:2d} — loss: {loss:.4f}, acc: {acc*100:5.2f}%")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
    
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2
        )
    
    @classmethod
    def load(cls, path: str) -> "MLP":
        npz = np.load(path)
        model = cls()
        model.W1, model.b1 = npz["W1"], npz["b1"]
        model.W2, model.b2 = npz["W2"], npz["b2"]
        return model


if __name__ == "__main__":
    print("Fetching openml mnist_784 dataset.")
    mnist = fetch_openml("mnist_784", version=1)
    
    X = mnist.data.to_numpy(dtype=np.float32)  # Shape (70000, 784)
    y = mnist.target.to_numpy(dtype=int)       # Shape (70000,)
    
    # Normalize the gray scale to 0-1
    X /= 255
    
    input_size = X.shape[1]
    hidden_size = 128
    output_size = len(np.unique(y))            # Shape (70000, 10)
    
    Y = np.eye(output_size)[y]  # Convert y into one-hot.
    
    # Split into 60k train set and 10k test.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/7, random_state=42
    )
    
    print("Initializing and fitting model.")
    net = MLP(input_size, hidden_size, output_size)
    net.fit(X_train, Y_train, epochs=10, lr=0.1, batch_size=128)
    
    test_acc = float(np.mean(net.predict(X_test) == np.argmax(Y_test, axis=1)))
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    
    save_file = "mlp_mnist.npz"
    net.save(save_file)
    print(f"Model saved to {save_file}")
