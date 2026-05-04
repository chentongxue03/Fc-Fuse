# Fc-Fuse

Fc-Fuse 是一个基于 TensorFlow/Keras 的网络流量分类实验项目。项目将网络流量样本拆分为统计特征、包长序列特征和载荷字节序列特征三类输入，并通过多分支神经网络进行特征提取、融合和分类。

当前代码主要用于复现实验训练流程，包括数据预处理、模型构建、训练、早停保存、测试集评估、分类报告和混淆矩阵输出。

## 项目结构

```text
Fc-Fuse/
|-- file/                  # 本地数据目录，CSV 数据集不提交到 GitHub
|-- main/
|   |-- data_pressor.py    # 数据读取、标签编码、特征处理和训练/测试集划分
|   |-- evaluate.py        # 测试集评估、分类报告和混淆矩阵生成
|   |-- loss.py            # 分类损失和跨模态对齐损失
|   |-- main.py            # 训练入口
|   |-- model.py           # STATS、PKT、PYL 分支和融合分类模型
|   `-- train.py           # 单步训练逻辑
|-- requirements/
|   `-- requirements.txt   # Python 依赖
|-- .gitignore
`-- README.md
```

## 环境依赖

建议先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements/requirements.txt
```

主要依赖包括：

- `tensorflow`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## 数据集说明

本仓库不包含原始 CSV 数据文件。原因是项目中的数据集文件体积较大，不适合直接提交到普通 GitHub 仓库：

- `file/botiot_last_datasets.csv`：当前默认训练数据文件
- `file/CIC_bl_ordered.csv`：本地保留的另一个 CSV 数据文件，可作为替换实验数据使用

这些文件已经通过 `.gitignore` 中的规则 `file/*.csv` 排除，因此不会被 `git add .` 或 `git push` 上传。运行项目时，需要在本地手动把对应 CSV 放到 `file/` 目录下。

当前训练入口 `main/main.py` 默认读取：

```text
../file/botiot_last_datasets.csv
```

如果要使用 `CIC_bl_ordered.csv` 或其他同格式数据集，需要修改 `main/main.py` 中的 `csv_path`：

```python
csv_path = "../file/botiot_last_datasets.csv"
```

## CSV 字段要求

`main/data_pressor.py` 对 CSV 的字段有固定假设，数据文件至少需要包含以下列：

| 字段名 | 作用 | 处理方式 |
| --- | --- | --- |
| `category` | 分类标签 | 使用 `LabelEncoder` 编码为类别编号 |
| `pktlen` | 包长序列特征 | 按英文逗号分隔，转为整数序列，取绝对值后作为 PKT 分支输入 |
| `fwd_payload` | 正向载荷 | 按十六进制字符串解析为字节序列 |
| `bwd_payload` | 反向载荷 | 按十六进制字符串解析为字节序列 |

以下字段会被排除，不作为统计特征输入：

```text
Uuid, pktlen, fwd_payload, bwd_payload, category, subclass
```

除了上述排除字段外，CSV 中的其他列会被视为统计特征：

1. 缺失值使用 `0` 填充；
2. 通过 `StandardScaler` 标准化；
3. 输入到 STATS 分支进行特征提取。

`pktlen` 的示例格式：

```text
60,60,-52,1500,40
```

`fwd_payload` 和 `bwd_payload` 的示例格式：

```text
4500003c1c4640004006
```

如果载荷为空，代码中使用 `-` 或空值时会按全 `0` 序列处理。

## 模型输入与处理流程

项目将每条流量样本处理为三类输入：

1. **STATS 特征**：CSV 中除标签、载荷、包长等字段之外的数值统计特征；
2. **PKT 特征**：由 `pktlen` 解析得到的包长序列，最大长度默认限制为 `20`；
3. **PYL 特征**：由 `fwd_payload` 和 `bwd_payload` 解析得到的字节序列，单向载荷最大长度默认限制为 `50`，拼接后作为 PYL 输入。

模型结构包括：

- `STATS_NN`：全连接网络，用于统计特征编码；
- `PKT_NN`：Embedding + BiLSTM，用于包长序列编码；
- `PYL_NN`：Embedding + Conv1D + GlobalMaxPooling，用于载荷字节序列编码；
- `FusionMLP`：融合三类特征；
- `ClassifierModel`：输出最终分类结果。

## 运行方式

进入训练入口所在目录：

```bash
cd main
python main.py
```

训练参数位于 `main/main.py` 的 `main()` 函数中，可按实验需要修改：

```python
epochs = 50
batch_size = 256
alpha = 1.0
beta = 0.5
margin = 1.0
learning_rate = 1e-3
patience = 5
```

## 输出结果

训练过程中会输出每轮损失和准确率。验证集准确率提升时，模型会保存到：

```text
main/logs/<timestamp>/best_model_epoch*_acc*
```

训练结束后会加载最佳模型并在测试集上评估，主要输出包括：

- 测试集准确率；
- 分类报告 `classification_report.txt`；
- 混淆矩阵图 `confusion_matrix.png`；
- 训练损失曲线 `logs/<timestamp>/training_losses.png`。

注意：`classification_report.txt` 和 `confusion_matrix.png` 当前会生成在运行命令所在目录下。如果从 `main/` 目录运行，它们会出现在 `main/` 目录中。

## GitHub 上传说明

本仓库只提交代码、依赖文件和项目说明，不提交大型 CSV 数据集和训练输出。已忽略的内容包括：

- `file/*.csv`
- `logs/`
- `outputs/`
- `runs/`
- `checkpoints/`
- 常见模型权重文件，如 `*.pt`、`*.pth`、`*.h5`、`*.keras`

如果后续需要共享数据集，建议将 CSV 上传到网盘、Kaggle Dataset、GitHub Release 或使用 Git LFS，再在 README 中补充下载链接和数据来源说明。
