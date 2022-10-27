Train_data.csv	全部训练集数据未归一化(包含y值）
Test_data.csv 	全部测试集数据未归一化（包含y值）
Val_data_X.csv	全部验证集数据未归一化（没有y值）
Train_data_I.xlsx/.csv	模态I训练集未归一化
Test_data_I.xlsx/.csv	模态I测试集未归一化
Train_data_I_mm.xlsx	模态I训练集归一化
Test_data_I_mm.xlsx	模态I测试集归一化
Train_data_modeI_chamber4.csv	模态I腔室4训练集未归一化
Test_data_modeI_chamber4.csv	模态I腔室4测试集未归一化
Train_data_modeI_chamber4_mm.csv	模态I腔室4训练集归一化
Test_data_modeI_chamber4_mm.csv	模态I腔室4测试集归一化
X_train_r_modeI_chamber4_mm.npy	模态I腔室4训练集归一化时间补齐，shape=(798, 263, 19)
y_train_modeI_chamber4_mm.npy	模态I腔室4训练集MRR，shape=(798, 1)
X_test_r_modeI_chamber4_mm.npy	模态I腔室4测试集归一化时间补齐，shape=(165, 263, 19)
y_test_modeI_chamber4_mm.npy	模态I腔室4测试集MRR，shape=(165, 1)
wafer_id_train_modeI_chamber4_mm.pkl	模态I腔室4训练集归一化时间补齐后的waferid顺序，从小到大顺序，和X_train_r_modeI_chamber4_mm.npy对应
wafer_id_test_modeI_chamber4_mm.pkl	模态I腔室4测试集归一化时间补齐后的waferid顺序，从小到大顺序，和X_test_r_modeI_chamber4_mm.npy对应
train_x_mean_modeI_chamber4_mm.csv	模态I腔室4训练集归一化提取平均值特征后的feature,shape=(798,19)
train_y_mean_modeI_chamber4_mm.csv	模态I腔室4训练集归一化提取平均值特征后的对应的y,shape=(798,)
test_x_mean_modeI_chamber4_mm.csv	模态I腔室4测试集归一化提取平均值特征后的feature,shape=(165,19)
test_y_mean_modeI_chamber4_mm.csv	模态I腔室4测试集归一化提取平均值特征后的对应的y,shape=(798,)
lstm2_output_train.npy	模态I腔室4训练集归一化时间补齐之后LSTM模型第2个隐层的输出shape=（798， 40）
lstm2_output_test.npy	模态I腔室4测试集归一化时间补齐之后LSTM模型第2个隐层的输出shape=（165， 40）
biLstm2_output_train.npy	模态I腔室4训练集归一化时间补齐之后biLSTM模型第2个隐层的输出shape=（798， 80）
biLstm2_output_test.npy	模态I腔室4测试集归一化时间补齐之后biLSTM模型第2个隐层的输出shape=（165， 80）
conv1d_output_train.npy	模态I腔室4训练集归一化时间补齐之后conv1d模型特征输出shape=（798， 64）
conv1d_output_test.npy	模态I腔室4测试集归一化时间补齐之后conv1d模型特征输出shape=（165， 64）
Train_data_for_OD.csv	用于异常检测（Outlier Detection)的训练数据，归一化，包括模态IChamber4的数据和MRR值达到4000+的数据点，总共shape=(149023, 26)

c_center.pickle是5个时段的聚类中心，模态I腔室4

df_stable_seg_update_train.csv	基于RETAINER_RING_PRESSURE	的时段划分结果，基于模态I腔室4训练集归一化之后的数据，index='WAFER_ID', columns=['n_stable', 'b_stable', 'b_stable_update']
# n_stable_segment表示时段的个数
# b_stable表示时段的左起点和右终点，例如[1,2,4,5]表示2个时段分别是[1,2)和[4,5)
# b_stable_update表示分割点，例如[1,4,7]表示2个时段分别是[1,4)和[4,7)
df_stable_seg_update_test.csv 	文件格式同df_stable_seg_update_train，基于RETAINER_RING_PRESSURE	的时段划分结果，基于模态I训练集归一化之后的数据

Train_seg.dict	字典结构，key=['seg_0', 'seg_1', 'seg_2', 'seg_3', 'seg_4']，基于模态I腔室4训练集归一化之后的数据根据df_stable_seg_update.csv分段后的结果，以seg_0为例，元素仍然是一个字典，字典的key是798个WAFER_ID
Test_seg.dict 字典结构，和Train_seg.dict为一组，注意AVERAGE_RATE也保存进来了

Train_seg_X_r.dict 字典类型，模态I腔室4的5个稳定时段的研磨速率值，key=['seg_0', 'seg_1', 'seg_2', 'seg_3', 'seg_4']，元素是元组类型，例如（797,37,19)只使用了19个变量，cols_X = cols_X = ['USAGE_OF_BACKING_FILM', 'USAGE_OF_DRESSER',
       'USAGE_OF_POLISHING_TABLE', 'USAGE_OF_DRESSER_TABLE',
       'PRESSURIZED_CHAMBER_PRESSURE', 'MAIN_OUTER_AIR_BAG_PRESSURE',
       'CENTER_AIR_BAG_PRESSURE', 'RETAINER_RING_PRESSURE',
       'RIPPLE_AIR_BAG_PRESSURE', 'USAGE_OF_MEMBRANE',
       'USAGE_OF_PRESSURIZED_SHEET', 'SLURRY_FLOW_LINE_A',
       'SLURRY_FLOW_LINE_B', 'SLURRY_FLOW_LINE_C', 'WAFER_ROTATION',
       'STAGE_ROTATION', 'HEAD_ROTATION', 'DRESSING_WATER_STATUS',
       'EDGE_AIR_BAG_PRESSURE']
Train_seg_Y_r.dict 字典类型，5个稳定时段的研磨速率值，key=['seg_0', 'seg_1', 'seg_2', 'seg_3', 'seg_4']，元素是list类型

Test_seg_X_r.dict 测试集部分，和Train_seg_X_r.dict类型一致
Test_seg_Y_r.dict 测试集部分，和Train_seg_Y_r.dict类型一致

conv1d_output_train_segment_{i}.array 5个时段的训练集数据采用conv1d模型训练后的特征
conv1d_output_test_segment_{i}.array 5个时段的测试集数据采用conv1d模型训练后的特征
对应的conv1d模型保存在/save_model/segment_{I}_model_conv1d.h5

biLSTM_output_train_segment_{i}.array 5个时段的训练集数据采用biLSTM模型训练后的特征
biLSTM_output_test_segment_{i}.array 5个时段的测试集数据采用biLSTM模型训练后的特征
对应的biLSTM模型保存在/save_model/segment_{I}_model_biLSTM.h5

lstm_output_train_segment_{i}.array 5个时段的训练集数据采用lstm模型训练后的特征
lstm_output_test_segment_{i}.array 5个时段的测试集数据采用lstm模型训练后的特征
对应的lstm模型保存在/save_model/segment_{I}_model_lstm.h5

wafer_id_train_modeI_chamber4.list 是训练集model I腔室4的WAFER_ID，顺序与时段划分结果的样本顺序不一致，总共批次是798个，时段划分只用到了797个批次
wafer_id_train_seg.list 是训练集model I腔室4的WAFER_ID，顺序与时段划分结果的样本顺序不一致，时段划分只用到了797个批次，wafer_id=4167773580时分不出5个时段所以删除这个批次
wafer_id_test_modeI_chabmer4.list	是测试集model I腔室4的WAFER_ID，顺序与时段划分结果的样本顺序一致

conv1d_sae_features_train.csv	采用conv1d提取的时段特征输入到sae模型，提取压缩特征，训练集部分
conv1d_sae_features_test.csv	采用conv1d提取的时段特征输入到sae模型，提取压缩特征，测试集部分
biLSTM_sae_features_train.csv	采用biLSTM提取的时段特征输入到sae模型，提取压缩特征，训练集部分
biLSTM_sae_features_test.csv	采用biLSTM提取的时段特征输入到sae模型，提取压缩特征，测试集部分
LSTM_sae_features_train.csv	采用LSTM提取的时段特征输入到sae模型，提取压缩特征，训练集部分
LSTM_sae_features_test.csv	采用LSTM提取的时段特征输入到sae模型，提取压缩特征，测试集部分

df_modeI_train.csv	模态I(CHAMBER=4/5/6,STAGE=A)训练集全部数据，带有原始的index，所以shape=(274138, 27)
df_modeII_train.csv	模态II(CHAMBER=4/5/6,STAGE=B)训练集全部数据，带有原始的index，所以shape=(295885, 27) 
df_modeIII_train.csv	模态III(CHAMBER=1/2/3,STAGE=A,删除'AVG_REMOVAL_RATE'>=1000的outlier)训练集全部数据，带有原始的index，所以shape=(101399, 27)

df_modeI_test.csv	模态I(CHAMBER=4/5/6,STAGE=A)测试集全部数据，带有原始的index，所以shape=(71801, 27)
df_modeII_test.csv	模态II(CHAMBER=4/5/6,STAGE=B)测试集全部数据，带有原始的index，所以shape=(64464, 27) 
df_modeIII_test.csv	模态III(CHAMBER=1/2/3,STAGE=A,删除'AVG_REMOVAL_RATE'>=1000的outlier)测试集全部数据，带有原始的index，所以shape=(19997, 26)
df_three_mode_train.csv	三个模态训练集全部数据，添加一个列Mode取值I/II/III，带有原始的index，所以shape=(671422, 28) ，其中模态III去除了噪声
df_three_mode_test.csv	三个模态测试集全部数据，添加一个列Mode取值I/II/III，带有原始的index，所以shape=(156262, 28)，其中模态III去除了噪声

features_modeI_train.csv	模态I训练集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(798, 98)
features_modeII_train.csv	模态II训练集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(815, 98)
features_modeIII_train.csv	模态III训练集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(364, 98)

features_modeI_test.csv	模态I测试集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(165, 98)
features_modeII_test.csv	模态II测试集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(186, 98)
features_modeIII_test.csv	模态III测试集统计特征(mean/std/median/sub/auc)，维度=19*5+3('WAFER_ID'/'STAGE'/'AVG_REMOVAL_RATE')=98，所以shape=(73, 98)

features_three_mode_train.csv	三个模态训练集统计特征，维度=19个变量*5个统计特征+4(['WAFER_ID', 'STAGE', 'AVG_REMOVAL_RATE', 'Mode'])=99，，所以shape=(1977, 98)
features_three_mode_test.csv	三个模态测试集统计特征，维度=19个变量*5个统计特征+4(['WAFER_ID', 'STAGE', 'AVG_REMOVAL_RATE', 'Mode'])=99，，所以shape=(424, 98)

X_train_r_three_mode.npy	三个模态训练集三维批次补0之后的数据，shape=(1963, 462, 19)，因为删除了批次采样个数超过500的样本，它们分别是[(-4228160596,  'II'),
            (-4226160408,  'II'),
            (-4019511766,  'II'),
            (   31494350,   'I'),
            (  665222804, 'III'),
            ( 1475739292,  'II'),
            ( 1836206950, 'III'),
            ( 2056207562, 'III'),
            ( 2056207566, 'III'),
            ( 3015014228,  'II'),
            ( 3021014294,  'II'),
            ( 4179773518,   'I'),
            ( 4181773366,   'I'),
            ( 4201773386,  'II')]
X_test_r_three_mode.npy	三个模态测试集三维批次补0之后的数据，shape=(1963, 462, 19)，因为删除了批次采样个数超过500的样本，它们分别是[(-4113511760, 'I'),
            ( 2925014266, 'I')]
y_train_three_mode.npy	与X_train_r_three_mode.npy顺序一一对应的AVG_REMOVAL_RATE
y_test_three_mode.npy	与X_test_r_three_mode.npy顺序一一对应的AVG_REMOVAL_RATE
wafer_id_mode_train_three_mode.pkl	与X_train_r_three_mode.npy顺序一一对应的(wafer_id, mode)
wafer_id_mode_test_three_mode.pkl	与X_test_r_three_mode.npy顺序一一对应的(wafer_id, mode)
cols_feature_three_mode.pkl	X_train_r_three_mode和X_test_r_three_mode对应的column名称

