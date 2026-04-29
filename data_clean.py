# step1_data_load.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 1.改为电脑内图片真实路径
DATA_PATH = r"E:\张博宇试作\CS600003\EuroSAT_RGB"
IMG_SIZE = 64  # 把所有图片统一缩放到64×64
CLASSES = ['AnnualCrop','Forest','HerbaceousVegetation','Highway',
           'Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']

def load_eurosat_data():
    images = []
    labels = []

    # 遍历每个类别文件夹
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_PATH, class_name)
        print(f"正在读取类别：{class_name} ({class_idx+1}/10)")

        # 遍历这个类别下的所有图片
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # 打开图片、缩放到统一尺寸、归一化
                img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0  # 把像素值从0-255缩到0-1
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"跳过损坏的图片：{img_path}，错误：{e}")

    # 转成numpy数组方便后续处理
    X = np.array(images)
    y = np.array(labels)
    print(f"\n✅ 读取完成！总图片数：{len(X)}")
    print(f"图片维度：{X.shape}（样本数, 高, 宽, 通道数）")
    return X, y

def split_data(X, y):
    """按 7:1.5:1.5 划分训练/验证/测试集"""
    # 先分 70% 训练集，剩下 30% 给验证+测试
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # 再把剩下的 30% 对半分，各15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # 把 64×64×3 的图片展平成一维向量（方便后面MLP输入）
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    print("\n📊 数据集划分结果：")
    print(f"训练集：X={X_train_flat.shape}, y={y_train.shape}")
    print(f"验证集：X={X_val_flat.shape}, y={y_val.shape}")
    print(f"测试集：X={X_test_flat.shape}, y={y_test.shape}")

    return X_train_flat, X_val_flat, X_test_flat, y_train, y_val, y_test

if __name__ == "__main__":
    print("=== 第一步：加载并预处理 EuroSAT 数据 ===")
    # 1. 读取所有图片
    X, y = load_eurosat_data()
    # 2. 划分训练/验证/测试集并展平
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    # 3. 保存成 .npz 文件
    np.savez(
        "eurosat_processed.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )
    print("\n💾 数据已保存为：eurosat_processed.npz")
