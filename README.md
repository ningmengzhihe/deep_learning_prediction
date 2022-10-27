## exp_basic_model
**试验机器学习模型预测**
part1全监督回归(1)线性模型：Ridge/Lasso/Elastic/BayesianRidge（2）KNN（3）树模型：DecisionTree/RandmoForest（4）集成模型：Adaboost/GradientBoostingDecisionTree/Xgboost/LightBGM（5）Stacking

part2半监督回归Coreg/Corareg

## exp_basic_model_for_mode_I
**特别针对模态I腔室4归一化后的预测**
模态I：腔室4、5、6，阶段A、MRR范围[50,90]、批次个数（训练集798、验证集185、测试集165）
提取统计特征：平均值
模型：（1）线性模型：LinearRegression/Lasso/Ridge/Elastic/BayesianRidge（2）KNN（3）SupportVector（4）树模型DecisionTree（5）集成树模型：RandomForest/ExtraTree（6）集成模型：AdaBoost/GBDT（GradientBoostingDecisionTree）/Xgboost/LightGBM

## exp_conv1d
**模态I CHAMBER=4 conv1 虚拟量测 conv1d model**

## exp_conv2d
**模态I CHAMBER=4 conv2d 虚拟量测 conv2d**

## exp_DeepLearning_multitimes
**汇总conv1d/conv2d/LSTM/RNN/GRU/BiLSTM模型，多次迭代取预测结果的平均值**

## exp_DeepLearning_stack
**CNN/RNN/Transformer features + regression**

## exp_modeI_time_segment_prediction
**时段划分模型预测MRR**
基于模态I腔室4的数据
模态I数据，分成了5个稳定时段，（1）在质量变量（MRR）的监督下，分时段训练各 LSTM 模型以提取各时段特征（2）训 练完成后，拼接各时段LSTM模型的输出向量作为SAE模型的输入（3）基于各时段 LSTM 模型的输出，SAE 首先对各自动编码器进行无监督的预训练，而后在质量变量监督下微调各层的权重和偏置

## exp_modeI_time_segment_with_single_variable
**模态I CHAMBER=4基于单变量轨迹的稳定时段划分**
来自文兰硕士论文（基于单变量轨迹的时段划分）：针对模态I/CHAMBER=4的单个变量（保持环压力）的时段划分
时段划分模型：文兰硕士论文第4章节

## exp_model_preprocessing
**Part 1 模态I腔室4数据的预处理过程**
筛选后的数据可用于非时段的深度学习回归模型，如conv1d/conv2d/SimpleRNN/LSTM/BiLSTM/GRU/conv1d+SAE/LSTM+SAE/BiLSTM+SAE

处理过程包括：训练集、测试集两个部分的X字段和y值做腔室筛选、批次的时间补齐
保存结果和waferid顺序

**Part 2 三个模态的数据全部未展开，用于深度学习模型训练**
模态I(CHAMBER=4/5/6,STAGE=A)

模态II(CHAMBER=4/5/6,STAGE=B)

模态III(CHAMBER=1/2/3,STAGE=A,删除'AVG_REMOVAL_RATE'<1000的outlier)

可用于的深度学习模型有：conv1d/conv2d/rnn/lstm/bilstm/gru

## exp_results_to_json
将预测结果保存成json格式文件

## exp_rnn
**模态I CHAMBER=4 RNN/LSTM/GRU/BiLSTM 虚拟量测**


## exp_three_modes_exp1
**数据集三个Mode的实验**
数据的输入特征提取的五个统计特征

实验（1）统计特征

实验（2）统计特征+model的LabelEncoder编码


## exp_three_modes_exp2
**三个模态数据在一起的预测实验**
实验（1）输入是统计特征，算法：机器学习算法

实验（2）输入是三维，深度学习算法

## exp_transformer_extract_features

## exp_transformer_for_cmp_wafer

## exp_transformer_for_cmp_wafer_without_validation

## exp_wide_deep
**Wide&Deep模型 模态I的Chamber4数据虚拟测量**
(1)提取批次间信息

(2)wide&deep模型，优化了代码的结构

(3)keras的Model有多个输入



























