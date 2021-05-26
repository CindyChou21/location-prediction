# location-prediction

## 说明

本项目以GeoLife数据集为例，进行位置预测。采用的方法是图嵌入模型(loc2vec/node2vec)和时序模型(LSTM/GRU)、图神经网络模型(GATs/GCN)和时序模型(LSTM/GRU)结合的融合算法。

GeoLife数据集下载：https://www.microsoft.com/en-us/download/details.aspx?id=52367

main.py ：包含数据处理、模型训练和评估过程，其中的参数自行设置和调节

GraphAttentionLayer.py ：GATs层

GraphConvolution.py ：GCN层

hmm_base.py ：基线模型hmm的训练和评估