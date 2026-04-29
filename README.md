Eurosat 遥感图像分类 MLP 实现
本项目使用**双隐藏层多层感知机（MLP）**完成 EuroSAT 遥感图像 10 分类任务，包含模型定义、训练、测试、可视化、模型保存与加载。
1，项目功能
- 支持自定义隐藏层大小
- 支持 ReLU / Sigmoid 激活函数切换
- 实现反向传播与梯度计算
- 学习率衰减 + L2 正则化
- 训练曲线、权重可视化、错例分析
- 模型权重保存与加载
- 输出测试准确率与混淆矩阵
2，环境依赖
Python 3.8+  
3，安装依赖库：
```bash
pip install numpy matplotlib scikit-learn
4，best_model.npz为模型训练结果,eurosat_processed.npz为图片处理结果，和model_testing放入同一文件夹运行即可测试
