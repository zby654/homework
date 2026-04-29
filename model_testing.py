# --------------- 无GUI报错 ---------------
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ====================== 1. 加载数据 ======================
def load_data():
    data = np.load(r"E:\张博宇试作\PythonProject1\eurosat_processed.npz")
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    X_train = (X_train - 0.5) * 2
    X_val = (X_val - 0.5) * 2
    X_test = (X_test - 0.5) * 2

    return X_train, X_val, X_test, data["y_train"], data["y_val"], data["y_test"]

# ====================== 2. MLP 模型结构 ======================
class MLP:
    def __init__(self, input_dim=12288, hidden1=512, hidden2=256, num_classes=10, activation="relu"):
        self.activation = activation
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2. / input_dim)
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2. / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, num_classes) * np.sqrt(2. / hidden2)
        self.b3 = np.zeros((1, num_classes))

    def relu(self, x): return np.maximum(0, x)
    def sigmoid(self, x): return 1 / (1 + np.exp(-x))
    def tanh(self, x): return np.tanh(x)

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        if self.activation == "relu": self.a1 = self.relu(self.z1)
        elif self.activation == "sigmoid": self.a1 = self.sigmoid(self.z1)
        else: self.a1 = self.tanh(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        if self.activation == "relu": self.a2 = self.relu(self.z2)
        elif self.activation == "sigmoid": self.a2 = self.sigmoid(self.z2)
        else: self.a2 = self.tanh(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        exps = np.exp(self.z3 - np.max(self.z3, axis=1, keepdims=True))
        self.out = exps / np.sum(exps, axis=1, keepdims=True)
        return self.out

# ====================== 加载模型 ======================
def load_model(path="best_model.npz"):
    data = np.load(path)
    weights = (data["W1"], data["b1"], data["W2"], data["b2"], data["W3"], data["b3"])
    print("✅ 模型加载成功！")
    return weights

# ====================== 测试 ======================
def test(model, X_test, y_test, weights):
    model.W1, model.b1, model.W2, model.b2, model.W3, model.b3 = weights
    out = model.forward(X_test)
    acc = np.mean(np.argmax(out, axis=1) == y_test)
    print(f"\n🎯 测试集准确率 = {acc:.4f}")
    cm = confusion_matrix(y_test, np.argmax(out, axis=1))
    print("混淆矩阵：")
    print(cm)
    return np.argmax(out, axis=1)

# ====================== 画图 ======================
def plot_curve(train_acc, val_acc, train_loss, val_loss):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.savefig("training_curve.png", dpi=300)
    plt.close()
    print("✅ 训练曲线已保存")

def visualize_weights(model):
    W = model.W1
    plt.figure(figsize=(10,8))
    for i in range(16):
        w = W[:,i].reshape(64,64,3)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        plt.subplot(4,4,i+1)
        plt.imshow(w)
        plt.axis('off')
    plt.savefig("weight_visualization.png", dpi=300)
    plt.close()
    print("✅ 权重图已保存")

def error_example(X_test, y_test, y_pred):
    idx = np.where(y_pred != y_test)[0]
    plt.figure(figsize=(12,6))
    for i in range(min(10, len(idx))):
        im = X_test[idx[i]].reshape(64,64,3)
        im = (im + 1) / 2
        plt.subplot(2,5,i+1)
        plt.imshow(im)
        plt.title(f"T:{y_test[idx[i]]} P:{y_pred[idx[i]]}")
        plt.axis('off')
    plt.savefig("error_examples.png", dpi=300)
    plt.close()
    print("✅ 错例分析图已保存")

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 1. 创建模型
    model = MLP(hidden1=512, hidden2=256, activation="relu")

    # 2. 加载你保存好的模型
    best_weights = load_model("best_model.npz")

    # 3. 测试！
    y_pred = test(model, X_test, y_test, best_weights)

    # 4. 保存所有图片（写报告用）
    visualize_weights(model)
    error_example(X_test, y_test, y_pred)

    print("\n🎉 全部完成！")