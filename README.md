# Precipitation-forecast-AGS-U-Net
Precipitation forecast AGS U-Net
# 代码说明

- FURENet文件夹
  - FURENet_raw.py：FURENet网络架构；
  - Attention_FURENet_Gaussian.py：论文创新算法AGS U-Net网络架构；
  - Attention_FURENet.py：FURENet + SelfAttention网络架构；
  - Attention_FURENet_meta.py：FURENet + SelfAttention + 元学习算法ANIL 网络架构；
  - Attention_FURENet_forecasting_rain.py：论文创新算法AGS U-Net预测降雨量网络架构；
  - FURENet.py：FURENet论文原始网络架构；
  - GaussianSmoothing.py：高斯滤波器平滑模块；
  - ResidualConv2d.py：残差卷积模块；
  - SEBlock.py：压缩激发模块；
  - SelfAttention.py：自注意力模块及注意力门模块。

- util文件夹
  - data_generate.ipynb：质量控制生成代码；
  - sample.csv：质量控制后筛选出的每场雨的有效帧数列表；
  - TrainingTemplate.py：神经网络训练模块类（可继承）；
  - TestingTemplate.py：神经网络测试模块类（可继承）；
- cpt文件夹：各神经网络预测结果及模型参数、训练节点保存，由于模型输出结果过大，附件提供[百度云下载链接](https://pan.baidu.com/s/1lom1nRTp0CXxMnaNNWw8kQ?pwd=551r)（提取码：551r）可供结果的下载，百度云文件包含FURENet_raw训练结果和AGS U-Net训练结果。
  - best.pth：保存深度学习模型训练中的最佳模型权重；
  - checkpoint.pth：模型的检查点文件，通常在模型训练的过程中周期性地保存，以便稍后恢复训练、评估或使用模型；
  - labels.pth：用于保存深度学习模型测试相关的标签数据；
  - outputs.pth：用于保存深度学习模型在输入上的预测或输出结果；
  - processed_outputs.pth：对深度学习模型输出进行了某种处理或后处理的结果，将outputs.pth输出的接近于0值的点还原为0；
- Furenet_train.py：神经网络训练代码。
- Furenet_train_meta.py：元学习ANIL算法训练代码。
- result_check_now.ipynb：结果可视化输出。
