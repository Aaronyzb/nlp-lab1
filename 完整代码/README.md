# 自然语言处理作业一：文本分类

这份作业包含六组实验：Binary BoW、Word Frequency、GloVe、两组自行训练的 Word2Vec，以及 BERT 微调。前五组用 Logistic Regression 分类，BERT 的输入长度为 64，训练 3 轮。

## 阅读与提交

- `report/实验报告.pdf`：实验报告。
- `report/实验报告.md`：可编辑的报告正文。
- `report/assets/`：九张数值结果图和两张日志关键输出图。
- `results/summary.csv`：六组模型的测试 Accuracy、Macro-F1 和流程耗时。
- `results/*_predictions.csv`：固定测试集逐样本预测，可独立核验分数。
- `results/*.json`：验证选参、逐类指标、混淆矩阵、训练配置与统计检查。
- `src/`、`tests/`、本说明和依赖文件：完整代码与运行说明。
- `HW-1/nyt.csv`、`HW-1/ag.csv`：原始数据，未修改。

测试集没有另存为 `test.csv`，而是由 `results/split_indices.json` 中的 `test` 列表指定。这些数字是 `HW-1/nyt.csv` 的数据行号，从 0 开始，不含表头，共 1152 条。六个模型都在这些新闻上评价，Accuracy 和 Macro-F1 汇总在 `results/summary.csv`。

## 环境安装

使用 **Python 3.12**，以下命令在项目根目录执行。

```powershell
py -3.12 -m venv .venv-reproduce
.\.venv-reproduce\Scripts\python.exe -m pip install -r requirements-classical.txt
.\.venv-reproduce\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-reproduce\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS 使用 `python3.12 -m venv .venv-reproduce` 和 `.venv-reproduce/bin/python`，其余命令相同。CUDA 仅用于兼容 NVIDIA 环境
传统模型使用 Python 3.12、NumPy 1.26.4、SciPy 1.12.0、Gensim 4.3.3 和 scikit-learn 1.6.1。BERT 在 RTX 4060 上运行。完整的环境版本和资源校验值见 `results/resource_provenance.json`。

## 下载资源

```powershell
.\.venv-reproduce\Scripts\python.exe src/download_resources.py all
```

GloVe 下载 `glove.6B.zip`其中的 `glove.6B.100d.txt`。BERT 使用 `google-bert/bert-base-uncased`，固定版本为 `86b5e0934494bd15c9632b12f734a8a67f723594`。

固定 BERT 版本是为了让后续运行使用相同的预训练权重。两个资源的 SHA-256 保存在 `results/resource_provenance.json`。

## 一键运行

```powershell
$env:OMP_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='8'
.\.venv-reproduce\Scripts\python.exe run_experiments.py
```

脚本依次下载资源、检查数据、训练六个模型、运行测试、计算统计结果，最后生成报告。资源已经下载时可加 `--skip-download`。如果传统模型和 BERT 使用两个环境，可用 `--classical-python` 和 `--bert-python` 指定各自的解释器。日志保存在 `logs/`，某一步失败时程序会停止。

## 单独运行各步骤

```powershell
.\.venv-reproduce\Scripts\python.exe src/common.py
.\.venv-reproduce\Scripts\python.exe src/train_classical.py --task bow
.\.venv-reproduce\Scripts\python.exe src/train_classical.py --task glove
.\.venv-reproduce\Scripts\python.exe src/train_classical.py --task w2v_ag
.\.venv-reproduce\Scripts\python.exe src/train_classical.py --task w2v_nyt
.\.venv-reproduce\Scripts\python.exe src/train_bert.py --device auto --batch-size 32
.\.venv-reproduce\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-reproduce\Scripts\python.exe src/analyze_results.py
.\.venv-reproduce\Scripts\python.exe src/report_evidence.py
.\.venv-reproduce\Scripts\python.exe src/report_charts.py
.\.venv-reproduce\Scripts\python.exe src/report_log_images.py
.\.venv-reproduce\Scripts\python.exe src/build_report.py
```

训练后单篇预测：

```powershell
.\.venv-reproduce\Scripts\python.exe src/predict.py --model bow_binary --text "The team won the football match."
.\.venv-reproduce\Scripts\python.exe src/predict.py --model bert --text "The company reported higher quarterly earnings."
```

模型参数可选 `bow_binary`、`bow_frequency`、`glove`、`w2v_ag`、`w2v_nyt`、`bert`。

## 实验设计与实现细节

1. 固定 `np.random.default_rng(42)` 随机排列，以 `floor(0.8N)` 和 `floor(0.9N)` 切分。训练/验证/测试数量为 9215/1152/1152，所有实验共用保存的原始 CSV 行号。
2. 词袋和 Word2Vec 使用同一正则分词器，统一处理大小写、缩写和数字。
3. Task 1 包含两种词袋表示。训练词表使用全部训练词项，不裁剪到固定 top-k。Binary 记录是否出现，Frequency 使用原始次数，均使用 Logistic Regression 分类。
4. GloVe 直接使用公开的固定 400,000 词词表；AG Word2Vec 只用 AG 文本，NYT Word2Vec 只用 NYT 训练文本。文档表示按有效词出现次数求均值，OOV 跳过，全 OOV 时使用零向量。标准化仅拟合 NYT 训练特征。
5. Logistic Regression 搜索 C={0.01,0.1,1,10} 与 class_weight={None,balanced}，以验证 Macro-F1 优先、Accuracy 次优选参。分类器只拟合训练标签，不合并验证集。
6. BERT 是指定预训练主体加分类头的全参数微调，采用 AdamW、学习率 2e-5、10% warmup、线性衰减、梯度裁剪 1.0。三轮全部完成，再评价验证最优 checkpoint 的测试表现。GPU 使用 FP16 AMP。
7. 词表、标准化和 NYT Word2Vec 在训练集上拟合，验证集用于模型选择；GloVe 使用公开预训练向量。
8. 保留题目指定随机划分中的原始重复文本。分层配对 bootstrap 仅用于最终结果的不确定性分析，不参与调参。

## 结果文件索引

| 文件 | 用途 |
| --- | --- |
| `data_audit.json` | 数据规模、分布、重复、文件哈希 |
| `split_indices.json` | 固定划分原始行号（从 0 开始，不含表头） |
| `bow_*.json`、`glove.json`、`w2v_*.json` | 八组验证候选、最终配置、测试指标 |
| `bert_history.json`、`bert.json` | 三轮训练记录、最优 epoch、测试指标 |
| `*_predictions.csv` | 逐样本预测与真实标签 |
| `statistical_analysis.json` | 多数类基线、95% 区间、配对差值 |
| `error_analysis.csv` | 全部错误的行号、类别和原文片段 |
| `top_binary_features.json` | 各类 Binary BoW 正向权重词 |
| `resource_provenance.json` | 资源来源、版本、SHA-256 与实际依赖 |
| `verification.json` | 最终完整性与 PDF 检查记录 |
| `report_evidence.json` | 重新核算六组预测和本次检查输出的来源 |

