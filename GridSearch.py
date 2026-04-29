import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from sklearn.metrics import confusion_matrix
import itertools


# ====================== 1. 加载数据 ======================
def load_data():
    data = np.load(r"E:\张博宇试作\PythonProject1\eurosat_processed.npz")
    return data["X_train"], data["X_val"], data["X_test"], data["y_train"], data["y_val"], data["y_test"]


# ====================== 2. 双隐藏层 + 多激活函数 MLP ======================
class MLP:
    def __init__(self, input_dim=12288, hidden1=256, hidden2=128, num_classes=10, activation="relu"):
        """
        双隐藏层版本
        支持自定义 hidden1, hidden2
        支持 relu / sigmoid / tanh 切换
        """
        self.activation = activation

        # 第一层
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden1))

        # 第二层（新增！）
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, hidden2))

        # 输出层
        self.W3 = np.random.randn(hidden2, num_classes) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros((1, num_classes))

    # 激活函数
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    # 激活函数导数
    def relu_grad(self, x):
        return (x > 0).astype(float)

    def sigmoid_grad(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh_grad(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, x):
        # 第一层
        self.z1 = x @ self.W1 + self.b1
        if self.activation == "relu":
            self.a1 = self.relu(self.z1)
        elif self.activation == "sigmoid":
            self.a1 = self.sigmoid(self.z1)
        elif self.activation == "tanh":
            self.a1 = self.tanh(self.z1)

        # 第二层
        self.z2 = self.a1 @ self.W2 + self.b2
        if self.activation == "relu":
            self.a2 = self.relu(self.z2)
        elif self.activation == "sigmoid":
            self.a2 = self.sigmoid(self.z2)
        elif self.activation == "tanh":
            self.a2 = self.tanh(self.z2)

        # 输出层
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = self.softmax(self.z3)
        return self.out

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    # 反向传播（双隐藏层 + L2 正则化）
    def backward(self, x, y, out, lr, weight_decay):
        n = y.shape[0]
        y_onehot = np.zeros_like(out)
        y_onehot[range(n), y] = 1

        # 输出层梯度
        dz3 = out - y_onehot
        dW3 = (self.a2.T @ dz3) / n + weight_decay * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True) / n

        # 第二层隐藏层梯度
        da2 = dz3 @ self.W3.T
        if self.activation == "relu":
            dz2 = da2 * self.relu_grad(self.z2)
        elif self.activation == "sigmoid":
            dz2 = da2 * self.sigmoid_grad(self.z2)
        elif self.activation == "tanh":
            dz2 = da2 * self.tanh_grad(self.z2)

        dW2 = (self.a1.T @ dz2) / n + weight_decay * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / n

        # 第一层隐藏层梯度
        da1 = dz2 @ self.W2.T
        if self.activation == "relu":
            dz1 = da1 * self.relu_grad(self.z1)
        elif self.activation == "sigmoid":
            dz1 = da1 * self.sigmoid_grad(self.z1)
        elif self.activation == "tanh":
            dz1 = da1 * self.tanh_grad(self.z1)

        dW1 = (x.T @ dz1) / n + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / n

        # SGD 更新
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3


# ====================== 3. 训练（含学习率衰减 + 自动保存最优模型） ======================
def train(model, X_train, y_train, X_val, y_val, epochs, lr, weight_decay, lr_decay):
    best_val_acc = 0
    best_weights = None

    for epoch in range(epochs):
        out = model.forward(X_train)
        model.backward(X_train, y_train, out, lr, weight_decay)

        # 学习率衰减
        lr *= lr_decay

        # 评估
        train_acc = np.mean(np.argmax(out, axis=1) == y_train)
        val_out = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_out, axis=1) == y_val)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = (model.W1.copy(), model.b1.copy(),
                            model.W2.copy(), model.b2.copy(),
                            model.W3.copy(), model.b3.copy())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1} | LR={lr:.4f} | Train={train_acc:.3f} | Val={val_acc:.3f}")

    print(f"\n最佳验证准确率 = {best_val_acc:.4f}")
    return best_weights


# ====================== 4. 测试 + 混淆矩阵 ======================
def test(model, X_test, y_test, best_weights):
    model.W1, model.b1, model.W2, model.b2, model.W3, model.b3 = best_weights
    out = model.forward(X_test)
    acc = np.mean(np.argmax(out, axis=1) == y_test)
    print(f"\n🎯 测试集准确率 = {acc:.4f}")

    # 混淆矩阵
    y_pred = np.argmax(out, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 混淆矩阵：")
    print(cm)


# ====================== 5. 网格搜索超参数 ======================
def grid_search(X_train, y_train, X_val, y_val):
    lr_list = [0.01, 0.05, 0.1]
    hidden_list = [(128, 64), (256, 128), (512, 256)]  # 双隐藏层参数
    wd_list = [0.0001, 0.001, 0.01]

    best_acc = 0
    best_params = None

    for lr, (h1, h2), wd in itertools.product(lr_list, hidden_list, wd_list):
        model = MLP(hidden1=h1, hidden2=h2, activation="relu")
        weights = train(model, X_train, y_train, X_val, y_val, epochs=300, lr=lr, weight_decay=wd, lr_decay=0.995)
        model.W1, model.b1, model.W2, model.b2, model.W3, model.b3 = weights
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)

        print(f"\n✅ LR={lr}, Hidden=({h1},{h2}), WD={wd} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = (lr, h1, h2, wd)

    print("\n🏆 最优超参数：", best_params)
    return best_params
