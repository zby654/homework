# 解决 OMP 报错
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 统一后端，杜绝Qt报错
plt.switch_backend("TkAgg")

# ====================== 1. 加载数据 + 归一化 ======================
def load_data():
    data = np.load(r"E:\张博宇试作\PythonProject1\eurosat_processed.npz")
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    X_train = (X_train - 0.5) * 2
    X_val = (X_val - 0.5) * 2
    X_test = (X_test - 0.5) * 2

    return X_train, X_val, X_test, data["y_train"], data["y_val"], data["y_test"]

# ====================== 2. 双隐藏层 MLP ======================
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
    def relu_grad(self, x): return (x > 0).astype(float)
    def sigmoid_grad(self, x): return self.sigmoid(x) * (1 - self.sigmoid(x))
    def tanh_grad(self, x): return 1 - np.tanh(x)**2

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        if self.activation == "relu":
            self.a1 = self.relu(self.z1)
        elif self.activation == "sigmoid":
            self.a1 = self.sigmoid(self.z1)
        else:
            self.a1 = self.tanh(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        if self.activation == "relu":
            self.a2 = self.relu(self.z2)
        elif self.activation == "sigmoid":
            self.a2 = self.sigmoid(self.z2)
        else:
            self.a2 = self.tanh(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = self.softmax(self.z3)
        return self.out

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, x, y, out, lr, weight_decay):
        n = y.shape[0]
        y_onehot = np.zeros_like(out)
        y_onehot[range(n), y] = 1

        dz3 = out - y_onehot
        dW3 = (self.a2.T @ dz3)/n + weight_decay * self.W3
        db3 = np.sum(dz3, axis=0, keepdims=True)/n

        da2 = dz3 @ self.W3.T
        if self.activation == "relu":
            dz2 = da2 * self.relu_grad(self.z2)
        elif self.activation == "sigmoid":
            dz2 = da2 * self.sigmoid_grad(self.z2)
        else:
            dz2 = da2 * self.tanh_grad(self.z2)

        dW2 = (self.a1.T @ dz2)/n + weight_decay * self.W2
        db2 = np.sum(dz2, axis=0, keepdims=True)/n

        da1 = dz2 @ self.W2.T
        if self.activation == "relu":
            dz1 = da1 * self.relu_grad(self.z1)
        elif self.activation == "sigmoid":
            dz1 = da1 * self.sigmoid_grad(self.z1)
        else:
            dz1 = da1 * self.tanh_grad(self.z1)

        dW1 = (x.T @ dz1)/n + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0, keepdims=True)/n

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

# ====================== 训练（保存日志用于画图） ======================
def train(model, X_train, y_train, X_val, y_val, epochs, lr, weight_decay, lr_decay):
    best_val_acc = 0
    best_weights = None
    train_acc_hist, val_acc_hist = [], []
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(epochs):
        out = model.forward(X_train)
        # 交叉熵loss
        train_loss = -np.mean(np.log(out[range(len(y_train)), y_train] + 1e-8))
        model.backward(X_train, y_train, out, lr, weight_decay)

        if (epoch + 1) % 200 == 0:
            lr *= lr_decay

        val_out = model.forward(X_val)
        val_loss = -np.mean(np.log(val_out[range(len(y_val)), y_val] + 1e-8))
        train_acc = np.mean(np.argmax(out, axis=1) == y_train)
        val_acc = np.mean(np.argmax(val_out, axis=1) == y_val)

        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = (model.W1.copy(), model.b1.copy(),
                           model.W2.copy(), model.b2.copy(),
                           model.W3.copy(), model.b3.copy())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} | LR={lr:.4f} | Train={train_acc:.3f} | Val={val_acc:.3f}")

    print(f"最佳验证准确率 = {best_val_acc:.4f}")
    return best_weights, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist

# ====================== 测试 ======================
def test(model, X_test, y_test, best_weights):
    model.W1, model.b1, model.W2, model.b2, model.W3, model.b3 = best_weights
    out = model.forward(X_test)
    acc = np.mean(np.argmax(out, axis=1) == y_test)
    print(f"\n🎯 测试集准确率 = {acc:.4f}")
    cm = confusion_matrix(y_test, np.argmax(out, axis=1))
    print("\n📊 混淆矩阵：")
    print(cm)
    return np.argmax(out, axis=1)

# ====================== 模型保存 ======================
def save_model(weights, path="best_model.npz"):
    np.savez(path, W1=weights[0], b1=weights[1], W2=weights[2], b2=weights[3], W3=weights[4], b3=weights[5])
    print("✅ 模型已保存：best_model.npz")

# ====================== 1. 绘制Loss&Acc曲线（作业必交） ======================
def plot_curve(train_acc, val_acc, train_loss, val_loss_hist):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss_hist, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_acc_curve.png", dpi=300)
    plt.close()
    print("✅ 训练曲线已保存：loss_acc_curve.png")

# ====================== 2. 第一层权重可视化（作业必交） ======================
def visualize_w1(model):
    W1 = model.W1
    plt.figure(figsize=(10,8))
    for i in range(16):
        w = W1[:, i].reshape(64,64,3)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        plt.subplot(4,4,i+1)
        plt.imshow(w)
        plt.axis("off")
    plt.suptitle("First Layer Weight Visualization")
    plt.savefig("weight_vis.png", dpi=300)
    plt.close()
    print("✅ 权重可视化图已保存：weight_vis.png")

# ====================== 3. 错例分析图（作业必交） ======================
def error_vis(X_test, y_test, y_pred):
    wrong_idx = np.where(y_pred != y_test)[0]
    plt.figure(figsize=(12,6))
    for i, idx in enumerate(wrong_idx[:10]):
        img = X_test[idx].reshape(64,64,3)
        img = (img + 1) / 2
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
        plt.axis("off")
    plt.suptitle("Error Analysis Samples")
    plt.savefig("error_sample.png", dpi=300)
    plt.close()
    print("✅ 错例分析图已保存：error_sample.png")

# ====================== 主程序 ======================
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    print("\n=== 开始最终训练 ===")
    model = MLP(hidden1=512, hidden2=256, activation="relu")
    best_weights, tracc, valacc, trloss, valloss = train(
        model, X_train, y_train, X_val, y_val,
        epochs=3000, lr=0.04, weight_decay=3e-4, lr_decay=0.95
    )

    y_pred = test(model, X_test, y_test, best_weights)
    save_model(best_weights)

    # 全部画图并保存
    plot_curve(tracc, valacc, trloss, valloss)
    visualize_w1(model)
    error_vis(X_test, y_test, y_pred)