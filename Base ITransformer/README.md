# 修改部分
基于官方给出的ITransformer进行修改

## 模型模块修改

### 位置编
对于原始的嵌入层进行了改进，添加了位置编码部分。考虑到数据是围绕当前站点的九宫形式，所以采用类似的相对位置编码结构。
```python
 [
    [2, 1, 2],
    [1, 0, 1],
    [2, 1, 2]
]
```
使用方法：
```python
python -u train.py ... --is_positional_embedding 1 ...
```

### RNN嵌入层
添加了可选的RNN嵌入层，以代替原始的线性嵌入层。

使用了双向GRU或LSTM进行嵌入，并在解码层使用对应的单向RNN模型进行。

使用方法：
```python
python -u train.py ... --time_embedding lstm ...
```

### 多任务学习
对于编码后的特征，添加额外分支，用于单独预测 **温度/风速** 部分。

首先将特征维度进行融合（目前使用线性层融合），得到融合特征。然后使用解码层进行预测。目前解码层与原始分支共享参数。

使用方法：
```python
python -u train.py ... --mode 1 ...
```

### 记忆模块
使用 **MPCount** 模型提出的记忆模块，来增强跨域泛化能力。

使用方法：
```python
python -u train.py ... --use_mem 1 --mem_size 512 ...
```

## DataSet模块修改

### 添加验证集
随机从所有站点中进行抽取作为验证集。

使用方法：
```python
python -u train.py ... --val_ratio 0.15 --seed 42 ...
```

### 添加数据增强
添加了如下数据增强方法：
1. 加噪声
2. 时间偏移
3. 缩放
4. 缩放
5. 反转
6. 时间抖动
7. 随机裁剪
8. 平滑
9. 随机删除
10. 噪声混合

每一条数据都将按照设置的概率随机使用上述的一种数据增强方式

使用方法：
```python
python -u train.py ... --augment_prob 1e-5 ...
```

## 整体框架
添加Debug

设置为debug模式，只载入5个站点数据进行测试，并且不保存模型。

使用方法：
```python
python -u train.py ... --debug 1 ...
```

## 推理部分

添加了模型集成部分，按照预设的权重对结果进行加权平均。模型权重名称必须以 **arg{num}** 来结尾。代码将加载目录下所有权重。对于推理结果，目前提供三种后处理方式：

1. 窗口加权平均
2. 卡尔曼滤波
3. 贝叶斯更新