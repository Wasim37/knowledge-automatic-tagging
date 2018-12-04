
# 知识点智能标注

## 1. 环境依赖

|环境/库|版本|
|:---------:|----------|
|conda|4.5.4|
|python|3.5.2|
|jupyter notebook|4.2.3|
|tensorflow|1.7.0|
|tensorboard|1.7.0|
|word2vec|0.9.4|
|numpy|1.14.0|
|pandas|0.23.1|
|matplotlib|1.5.3|


## 2. 文件结构

|- knowledge-automatic-tagging<br/>
|　　|- raw_data　　　　　　　　　   # 提取的题库原始数据<br/>
|　　|- data_process　　　　　　　   # 数据预处理代码<br/>
|　　|- data　　　　　　　　　　　   # 预处理得到的数据<br/>
|　　|- models　　　　　　　　　　  # 模型代码<br/>
|　　|　　|- wd-cnn-concat-1　　　	<br/>
|　　|　　|　　|- network.py　　　    # 定义网络结构<br/>
|　　|　　|　　|- train.py　　　　　   # 模型训练<br/>
|　　|　　|　　|- predict.py　　　　  # 验证集/测试集预测，生成概率矩阵<br/>
...<br/>
|　　|- ckpt　　　　　　　　　　　     # 保存训练好的模型<br/>
|　　|- summary　　　　　　　　　   # tensorboard 数据<br/>
|　　|- data_helpers.py　　　　　　   # 数据处理函数<br/>


## 3. 数据预处理
