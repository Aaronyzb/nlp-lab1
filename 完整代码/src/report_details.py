"""Report text and tables linked to actual six-model results and source files."""
import json
import numpy as np
import pandas as pd
from common import ROOT, LABELS
from analyze_results import NAMES, DISPLAY

def read(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))
def pct(value): return f'{value*100:.2f}%'

def build_sections(d):
    r={name:read(name+'.json') for name in NAMES}
    audit=read('data_audit.json'); stats=read('statistical_analysis.json'); top=read('top_binary_features.json')
    binary=r['bow_binary']; frequency=r['bow_frequency']; bert=r['bert']; w=d.width
    d.sections.append('实验概述')
    d.p('文本分类实验报告','title')
    d.p('自然语言处理作业一  实验日期 2026年10月8日','small')
    d.p('这次作业要完成的任务是：给定一篇新闻，预测它属于商业、政治还是体育。'
        '实验比较两种词袋表示、三种平均词向量表示，以及 BERT 微调。前五组使用 Logistic Regression，'
        '最后一组直接微调 BERT 的全部参数。所有结果均在同一批 1,152 篇 NYT 测试新闻上计算。')
    d.table(['任务','文本表示','分类器'],[
        ['Task 1','Binary BoW、Word Frequency','Logistic Regression'],
        ['Task 2','GloVe 100d、AG Word2Vec 100d、NYT Word2Vec 100d','Logistic Regression'],
        ['Task 3','BERT-base-uncased，max_length=64','BERT 三类别分类头']], [60,285,w-345])
    d.h('实验流程')
    d.table(['阶段','本次具体处理'],[
        ['准备数据','读取 CSV，检查空文本、标签和重复记录，固定一次 80%/10%/10% 划分。'],
        ['建立表示','训练词表或词向量，或者使用指定的公开词向量与 BERT tokenizer。'],
        ['训练与选参','只在训练集拟合参数，用验证集选 Logistic Regression 配置和 BERT checkpoint。'],
        ['最终评价','预测测试集，保存逐篇标签，再计算 Accuracy、Macro-F1 与混淆矩阵。']], [85,w-85])
    d.h('主要结果')
    d.p(f'Binary BoW 的测试 Accuracy 为 {pct(binary["test"]["accuracy"])}，Macro-F1 为 {pct(binary["test"]["macro_f1"])}。'
        f'Word Frequency 的 Accuracy 相同，Macro-F1 为 {pct(frequency["test"]["macro_f1"])}。'
        f'三组平均词向量中，NYT Word2Vec 最好；BERT 的测试 Macro-F1 为 {pct(bert["test"]["macro_f1"])}。'
        '后文把这些结果与词项覆盖率、类别表现、输入截断和具体错误对应起来分析。')
    d.p('图中 External 表示 AG 语料，Target 表示 NYT 训练语料。代码节选的源文件位置标在各段代码下方。','small')

    d.section('1 数据检查与类别分布')
    d.p('HW-1/nyt.csv 有 text 和 label 两列，共 11,519 篇新闻。HW-1/ag.csv 有 90,000 篇新闻，'
        '只有 text 列，用于训练外部语料的 Word2Vec。两份文件均没有空文本；AG 与 NYT 没有完全相同的文本。')
    d.table(['类别','全部样本','训练集','验证集','测试集'],[
        [label,sum(audit['split'][key]['class_counts'][label] for key in audit['split']),
         *[audit['split'][key]['class_counts'][label] for key in ['train','validation','test']]] for label in LABELS],
        [85,105,105,105,w-400])
    d.figure('class_distribution.png','各划分中三个类别的比例')
    d.p('体育新闻占全部样本的 75.00%，明显多于另外两类。直接打乱后按比例切分，三个子集的类别比例仍然接近。'
        '评价时同时观察 Macro-F1，可以更清楚地看到商业和政治两类的表现。')
    d.table(['检查项','实际结果'],[
        ['NYT 重复行','72 条；相同文本没有不同标签。'],['AG 重复行','29 条。'],
        ['训练与测试重复','共享 15 个不同的完全相同文本，后文另做过滤后的评价。'],
        ['普通分词后的长度',f"均值 {audit['token_lengths']['mean']:.1f}，中位数 {audit['token_lengths']['median']:.0f}，90% 分位数 {audit['token_lengths']['p90']:.0f}。"]], [135,w-135])
    d.p('数据检查结果保存在 results/data_audit.json。原始文件的 SHA-256 一并保存，后续可以确认数据文件没有被改动。','small')

    d.section('2 固定划分与分词代码')
    d.p('随机种子设为 42。先生成包含全部原始行号的随机排列，再在 80% 和 90% 的位置切开。'
        '训练、验证和测试集分别有 9,215、1,152、1,152 篇。测试行号从 0 开始，不包括 CSV 的表头。')
    d.excerpt('src/common.py','permutation =',10)
    d.p('permutation 只改变样本顺序，原始 CSV 不需要重写。保存 indices 后，词袋、词向量和 BERT 都通过同样的行号读取样本。'
        '程序会核对已有的划分文件，保证后续运行沿用相同的测试样本。')
    d.h('普通分词器')
    d.excerpt('src/common.py','TOKEN =',6)
    d.table(['输入或设置','作用'],[
        ['Apple apple, DON\'T 12.5!','得到 apple、apple、don\'t、12.5 四个词项。重复词不会在这里删除。'],
        ['转为小写','Apple 和 apple 共用同一个词项。'],
        ['保留撇号与数字','缩写和数字可以进入训练词表。'],
        ['不删停用词 不提取词干','保持基础词袋和平均词向量方法的处理一致。']], [155,w-155])
    d.p('本实验使用正则表达式进行英文分词，统一处理大小写、缩写和数字。遇到连字符时通常会分成多个词，'
        '不进行词形还原。BERT 的输入则由对应的 WordPiece tokenizer 处理。')

    d.section('3 Task 1 词袋向量如何构建')
    d.p('两种词袋共用一个训练词表，大小为 66,423。验证或测试中从未出现在训练集的词不会新增维度。'
        '例如，训练词表只有 apple 和 banana 时，下面的文档都被表示为两个维度。')
    d.table(['文档','Binary 向量','Frequency 向量'],[
        ['apple apple banana','[1, 1]','[2, 1]'],['banana','[0, 1]','[0, 1]'],
        ['apple unseen','[1, 0]','[1, 0]；unseen 未进入训练词表']], [150,125,w-275])
    d.p('Binary 的第 i 维为 1[c<sub>i</sub> &gt; 0]，只看出现与否；Frequency 的第 i 维为 c<sub>i</sub>，保留原始次数。'
        '这两种表示均不做长度归一化。')
    d.excerpt('src/train_classical.py','vectorizer = CountVectorizer',16)
    d.h('代码中的几个细节')
    d.p('CountVectorizer 在训练文本上建立词表，再用固定词表转换验证和测试文本。[4] '
        'token_pattern=None 表示使用自定义分词器；lowercase=False 是因为 tokenize 已经转为小写。')
    d.p('counts 为稀疏矩阵，只有非零词项需要存储。Binary 先复制 counts，再把非零值全部置为 1；'
        '这样可以同时保留原始计数矩阵，供 Frequency 实验使用。'
        'min_df=1 表示保留所有训练词项，没有人为截断到固定的 top-k 词表。')

    d.section('4 Logistic Regression 训练与模型选择')
    d.p('对文档向量 x，分类器为每个类别计算线性分数 z<sub>k</sub> = w<sub>k</sub><super>T</super>x + b<sub>k</sub>，'
        '通过 softmax 得到三个类别的概率，取概率最高的类别作为预测。训练使用 L2 正则；C 越小，正则约束越强。')
    d.table(['设置','取值或规则'],[
        ['分类器','LogisticRegression，lbfgs，max_iter=3000，tol=1e-4。'],
        ['正则参数 C','0.01、0.1、1、10。'],['类别权重','None 或 balanced，共八种候选组合。'],
        ['选参指标','验证集 Macro-F1 优先，其次 Accuracy；完全相同时保留先出现的组合。'],
        ['最终拟合范围','使用训练集拟合分类器，验证集用于选择配置。']], [130,w-130])
    d.excerpt('src/train_classical.py','for weight in [None',15)
    d.p('balanced 的类别权重为 N<sub>train</sub> / (3 × N<sub>k</sub>)。由于训练集体育样本多，'
        '商业和政治类样本会获得更大的权重。是否使用 balanced 由验证结果决定。')
    d.p('通过捕获 ConvergenceWarning 检查模型收敛情况。本次所有候选配置均正常收敛，'
        '迭代次数保存在各自的结果 JSON 中。')

    d.section('5 词袋模型的验证结果与选参依据')
    d.figure('validation_search.png','两种词袋模型的验证 Macro-F1 随 C 的变化')
    rows=[]
    for b,f in zip(binary['trials'],frequency['trials']):
        rows.append([b['C'],str(b['class_weight']),pct(b['validation_macro_f1']),pct(f['validation_macro_f1'])])
    d.table(['C','类别权重','Binary 验证 F1','Frequency 验证 F1'],rows,[55,100,170,w-325])
    d.p(f"Binary 选中的配置是 C={binary['selected']['C']}、class_weight={binary['selected']['class_weight']}；"
        f"Frequency 选中 C={frequency['selected']['C']}、class_weight={frequency['selected']['class_weight']}。"
        '两组配置分别对应上表中最高的验证集 Macro-F1。')
    d.p('两种向量的尺度不同：Binary 的非零值都为 1，Frequency 中较长文章或高频词会产生更大的值。'
        '从曲线看，两种表示的最佳 C 不同，增大 C 后分数也有下降的情况。')

    d.section('6 Task 2 词向量来源与 GloVe 加载')
    d.p('词袋为每个词分配一个独立维度，词向量则用固定长度的数值表示一个词。'
        'GloVe 根据大规模语料中的全局词共现统计学习向量；Word2Vec 则通过局部上下文训练。'
        '本题统一用 100 维词向量，再对每篇新闻中的有效词向量取平均。[1][2]')
    d.table(['实验','词向量来源','是否自行训练'],[
        ['GloVe','glove.6B.100d.txt；400,000 个词项','否，直接读取公开向量。'],
        ['AG Word2Vec','HW-1/ag.csv 的全部文本','是，只用 AG 文本。'],
        ['NYT Word2Vec','固定 NYT 训练集的 9,215 篇新闻','是，使用训练集文本。']], [110,245,w-355])
    d.h('读取 GloVe 文件')
    d.excerpt('src/train_classical.py',"vectors = {}",9)
    d.p('GloVe 文本文件的一行由一个词和 100 个浮点数构成。partition 取出词项，np.fromstring 将后面的数字转换为向量。'
        '读取时检查向量是否包含 100 个数，便于发现文件格式错误。')
    d.p('GloVe 使用公开预训练向量，NYT Word2Vec 则仅使用训练集语料，两者均不使用测试标签。')
    d.h('未登录词的处理')
    d.p('某个词不在对应词向量词表中时，跳过这个词。'
        '如果一篇文档没有任何有效词，返回全零向量，保证分类器仍能接收固定长度的输入。'
        '本次三组词向量的测试集都没有出现全零文档。')

    d.section('7 Word2Vec 的训练代码与语料规模')
    d.p('两组 Word2Vec 都使用 Gensim 的 Skip-gram 实现，训练参数相同，语料分别取 AG 文本和 NYT 训练文本。[2][5]')
    d.excerpt('src/train_classical.py',"if name == 'w2v_ag':",14)
    d.table(['参数','设置','含义'],[
        ['vector_size','100','词向量维度。'],['window','5','上下文窗口的最大范围。'],
        ['min_count','2','忽略语料中只出现一次的词。'],['sg / negative','1 / 5','采用 Skip-gram 和负采样。'],
        ['sample','0.001','对高频词进行下采样。'],['epochs / workers','10 / 1','训练十轮，单线程便于复现。'],
        ['seed / hashfxn','42 / stable_hash','固定随机初始化和词项哈希。']], [105,100,w-205],padding=4)
    d.figure('corpus_sizes.png','两份 Word2Vec 训练语料的文档数和词项出现次数')
    d.p(f"AG 的词表有 {r['w2v_ag']['metadata']['vocabulary_size']:,} 个词，NYT 训练语料词表有 {r['w2v_nyt']['metadata']['vocabulary_size']:,} 个词。"
        'AG 的文档数更多，但文本较短，总词项数约 356 万；NYT 训练集总词项数约 605 万。'
        'NYT 单篇新闻更长，因此文档数较少，总词项数却更高。')

    d.section('8 从词向量得到文档向量')
    d.function('src/train_classical.py','mean_embeddings')
    d.p('features 预先建立为“文档数 × 100”的零矩阵。对第 i 篇文档，valid 收集有向量的词项；'
        '同一个词出现两次，就加入两次向量。np.mean 沿 axis=0 求平均，得到一个 100 维行向量。')
    d.table(['二维演示 仅解释公式','数值'],[
        ['假设词向量','apple=[1,0]；banana=[0,1]。'],
        ['文档','apple apple banana unknown。'],
        ['有效词项数','3；unknown 不参与平均。'],
        ['平均向量','([1,0]+[1,0]+[0,1])/3 = [2/3,1/3]。']], [160,w-160],padding=4)
    d.p('上表用二维向量演示平均过程，实际实验使用 100 维向量。有效词项共有 3 个，因此求平均时除以 3。')
    d.figure('embedding_coverage.png','测试集有效词项覆盖率  按词项出现次数计算')
    d.p('覆盖率为“有向量的词项出现次数 / 全部词项出现次数”，重复词按出现次数计入。'
        '得到平均向量后，用 StandardScaler 调整各维度尺度；均值和标准差只在训练集拟合，'
        '再应用于验证和测试向量。三组文档向量使用相同的分类器选参规则。')

    d.section('9 三组平均词向量的测试结果')
    d.table(['方法','测试覆盖率','Accuracy','Macro-F1','C'],[
        [DISPLAY[NAMES.index(name)],pct(r[name]['metadata']['coverage']['test']['token_coverage']),
         pct(r[name]['test']['accuracy']),pct(r[name]['test']['macro_f1']),r[name]['selected']['C']]
        for name in ['glove','w2v_ag','w2v_nyt']], [120,100,95,95,w-410])
    d.p('三组模型最后都选择 class_weight=None。NYT Word2Vec 的 Macro-F1 比 AG Word2Vec 高 '
        f"{(r['w2v_nyt']['test']['macro_f1']-r['w2v_ag']['test']['macro_f1'])*100:.2f} 个百分点，测试覆盖率也更高。"
        'NYT 语料的主题与测试数据更接近，而且总词项数更多，可能是其分类效果更好的原因。')
    d.h('与词袋方法的对照')
    d.p('三组平均词向量的 Macro-F1 都低于两种词袋。平均操作把主题词、背景词和通用词混在同一个向量里，'
        '原本明显的主题线索可能被稀释。例如，词袋能单独保留 football 是否出现，平均词向量却需要把它和全文其他词一起平均。')
    d.table(['表示','文档特征维度','主要保留的信息'],[
        ['Binary / Frequency','66,423，稀疏','各词是否出现或出现多少次。'],
        ['平均词向量','100，稠密','有效词的分布式向量平均。'],
        ['BERT','分类头接收 768 维隐藏表示','在指定输入长度内编码上下文。']], [135,120,w-255])
    d.p('平均词向量把文档压缩为 100 维，减少了特征数量。词向量先通过上下文学习词语表示，'
        '再由 Logistic Regression 根据新闻标签学习分类边界。')
    d.h('如何保存模型以供预测')
    d.p('GloVe 查询表保存为 glove_vectors.joblib，Word2Vec 模型分别保存为 w2v_ag.model 和 w2v_nyt.model。'
        '对应分类器还保存训练集拟合的 StandardScaler。单篇预测时先按相同分词规则求均值，'
        '再用保存的 scaler 转换，最后调用分类器的 predict_proba。')

    d.section('10 Task 3 BERT 输入与分类头')
    d.p('使用 google-bert/bert-base-uncased 预训练模型。[3][6] 文本先由对应的 WordPiece tokenizer 转为 token ID，'
        '再输入预训练编码器和三类别分类头。普通分词器的词项与 WordPiece 子词不是同一种单位。')
    d.excerpt('src/train_bert.py','tokenizer = BertTokenizerFast',4)
    d.table(['输入项','作用'],[
        ['input_ids','词或子词及特殊标记的编号。'],['attention_mask','区分有效输入与补齐位置。'],
        ['token_type_ids','区分句段；本次按单段新闻输入。'],['labels','business=0、politics=1、sports=2。']], [140,w-140])
    d.function('src/train_bert.py','EncodedDataset')
    d.p('EncodedDataset 按保存的原始行号取出编码和标签，因此 BERT 与传统模型的划分完全一致。'
        'labels 使用 torch.long，对应三分类的交叉熵损失。训练 DataLoader 打乱批次顺序，验证和测试按固定顺序读取。')
    d.p('max_length=64 包含 [CLS] 和 [SEP]，正文最多约 62 个子词；短文本补齐，长文本截断。'
        f"测试集中 {pct(bert['metadata']['truncation']['test']['fraction_truncated'])} 的完整序列超过 64，"
        f"完整序列平均长度为 {bert['metadata']['truncation']['test']['mean_full_wordpiece_length']:.1f} 个 WordPiece。"
        '这说明几乎所有测试新闻都只输入了开头部分，和读取全文的词袋、平均词向量存在信息量差异。')

    d.section('11 BERT 全参数微调与优化步骤')
    d.excerpt('src/train_bert.py','model = BertForSequenceClassification.from_pretrained',5)
    d.p(f"模型有 {bert['metadata']['trainable_parameters']:,} 个可训练参数。分类头重新初始化，预训练主体也参与更新。"
        '训练前通过 requires_grad 检查全部参数的更新状态。')
    d.table(['项目','实际配置'],[
        ['训练轮数与 batch','3 epochs；batch size=32；每轮 288 个 batch，最后一批 31 条。'],
        ['优化器','AdamW，学习率 2e-5，weight decay=0.01。'],
        ['学习率计划','计划 864 个训练 batch，前 87 步 warmup，之后线性衰减。'],
        ['不做权重衰减的参数','bias 与 LayerNorm.weight。'],
        ['梯度处理','FP16 AMP，GradScaler，梯度范数裁剪为 1.0。'],
        ['运行设备','NVIDIA GeForce RTX 4060 Laptop GPU。']], [145,w-145],padding=4.5)
    d.excerpt('src/train_bert.py','scaler.scale(loss).backward()',9,before=2)
    d.p('每个 batch 先计算损失并反向传播。AMP 下先把梯度缩放值还原，再裁剪范数，随后更新参数。'
        '如果 GradScaler 因溢出跳过了参数更新，学习率调度也不向前走。训练模式打开 dropout，验证时切到 eval 模式。')

    d.section('12 BERT 三轮训练记录与模型选择')
    d.figure('learning_curve.png','三轮训练损失与验证集指标')
    d.table(['Epoch','训练损失','验证损失','验证 Accuracy','验证 Macro-F1'],[
        [row['epoch'],f"{row['train_loss']:.4f}",f"{row['validation_loss']:.4f}",
         pct(row['validation_accuracy']),pct(row['validation_macro_f1'])] for row in bert['history']],
        [50,95,95,125,w-365])
    d.figure('training_log_excerpt.png','训练日志的关键输出，已对原始日志进行排版整理。来源：logs/train_bert.log')
    d.p('第二轮验证 Macro-F1 为 95.36%，第三轮降到 95.08%；训练损失却继续下降，出现了过拟合的迹象。'
        f"训练完成三轮后，根据验证集 Macro-F1 选择第 {bert['selected_epoch']} 轮 checkpoint 进行测试。")
    d.p('日志节选包含每轮完成时的进度行和最终测试分数。完整日志中的模型加载、验证统计与 checkpoint 保存信息可在源文件中查看。','small')

    d.section('13 在测试集上计算 Accuracy 和 Macro-F1')
    d.p('测试评价发生在模型选择完成之后。传统模型调用 predict(features[split["test"]])；'
        'BERT 先加载验证集最优模型，再用 test DataLoader 预测。模型进入 eval 模式，并在 torch.no_grad 下执行，'
        '关闭 dropout 和梯度计算。')
    d.function('src/common.py','metrics')
    d.p('Accuracy 是预测正确的篇数除以测试篇数。对类别 k，F1<sub>k</sub> = '
        '2TP<sub>k</sub> / (2TP<sub>k</sub> + FP<sub>k</sub> + FN<sub>k</sub>)。Macro-F1 是三个类别 F1 的算术平均。'
        '计算时指定三个类别；某类的分母为零时，该类 F1 记为零。')
    d.figure('metric_output_excerpt.png','测试指标核算的关键输出，已对原始日志进行排版整理。来源：logs/report_metric_check.log')
    d.p('核算时逐项检查测试行号和真实标签，再把预测标签映射为 0、1、2，重算全部指标与混淆矩阵。'
        '六组结果与各模型 JSON 完全一致。')

    d.section('14 六组模型的最终测试结果')
    d.table(['方法','Accuracy','Macro-F1','错误数','流程秒数'],[
        [display,pct(r[name]['test']['accuracy']),pct(r[name]['test']['macro_f1']),
         1152-sum(r[name]['test']['confusion_matrix'][i][i] for i in range(3)),f"{r[name]['elapsed_seconds']:.1f}"]
        for name,display in zip(NAMES,DISPLAY)], [145,95,95,65,w-400])
    d.figure('model_scores.png','同一测试集上的 Accuracy 和 Macro-F1  横轴从零开始')
    gap=(binary['test']['macro_f1']-frequency['test']['macro_f1'])*100
    d.p(f'Binary 和 Frequency 都错分 12 篇，所以 Accuracy 相同；Binary 的 Macro-F1 高 {gap:.2f} 个百分点。'
        '三种平均词向量中，NYT Word2Vec 的两个指标都最高。BERT 错分 19 篇，比 Binary 多 7 篇，'
        'Binary 是本次测试中 Macro-F1 最高的模型。')
    d.p('时间包含表示构建、分类器训练和评价；Word2Vec 包含词向量训练，BERT 包含三轮训练、验证和模型保存。'
        '下载和环境安装不计入。由于传统模型和 BERT 使用不同计算设备，运行时间仅作为本次实验的参考。','small')

    d.section('15 各类别表现与完整混淆矩阵')
    d.figure('class_f1.png','六组模型的逐类 F1  颜色范围为 90% 至 100%')
    d.figure('all_confusions.png','六组测试混淆矩阵  行是真实类别  列是预测类别')
    d.p('所有模型的体育类 F1 都在 99% 以上，商业和政治类相对更难。Binary 正确识别了全部 152 篇政治新闻，'
        '同时把 6 篇商业和 2 篇体育新闻预测为政治；政治类 recall 为 100%，precision 为 95.00%。')
    d.p('Binary 和 BERT 的商业类正确数都为 129。BERT 比 Binary 多出的 7 个错误，来自政治类多错 4 篇和体育类多错 3 篇。'
        '矩阵数字为样本数量，颜色按真实类别所在行归一化。')

    d.section('16 指标手算与误分类分析')
    d.h('用 Binary 的商业类核对 F1')
    matrix=np.array(binary['test']['confusion_matrix']); tp=int(matrix[0,0]); fp=int(matrix[:,0].sum()-tp); fn=int(matrix[0].sum()-tp)
    d.table(['数量','值','对应含义'],[
        ['TP',tp,'商业新闻预测为商业。'],['FP',fp,'其他类别预测为商业。'],['FN',fn,'商业新闻预测成其他类别。']], [55,60,w-115])
    d.p(f'商业类 precision = {tp}/({tp}+{fp}) = {pct(tp/(tp+fp))}；'
        f'recall = {tp}/({tp}+{fn}) = {pct(tp/(tp+fn))}；'
        f'F1 = 2×{tp}/(2×{tp}+{fp}+{fn}) = {pct(2*tp/(2*tp+fp+fn))}。'
        '将商业、政治和体育三个 F1 求平均，得到 Binary 的 Macro-F1 97.79%。总正确数 1,140 除以 1,152，得到 Accuracy 98.96%。')
    d.h('具体错误记录')
    errors=pd.read_csv(ROOT/'results'/'error_analysis.csv')
    rows=[]
    for _,row in errors[errors.model=='bert'].head(4).iterrows():
        rows.append([int(row.row_id),row.true,row.predicted])
    d.table(['原始行号','数据标签','BERT 预测'],rows,[110,180,w-290])
    d.p('样本 8198 的数据标签是 sports，BERT 预测为 politics。文章讨论一场比赛失利之后发生的犯罪事件，'
        '开头既有体育背景，也有犯罪事件线索，主题存在交叉。BERT 只读取开头部分，可能更容易受到其中犯罪事件线索的影响。')
    d.h('高权重词项')
    d.table(['类别','Binary 正向权重最高的七个词'],[[label,', '.join(top[label][:7])] for label in LABELS],[80,w-80])
    d.p('football、league、team 与体育直接相关；washington、republican、obama 是政治类线索；company、market 与商业有关。'
        'mr 和 its 这样的通用词也在前列，说明分类器还利用了语料中的写作习惯。比较同一词在不同类别中的系数，'
        '可以进一步观察它对分类的作用。')

    d.section('17 重复文本与测试分数的波动')
    d.p('主实验按原始数据随机划分，保留重复记录。为检查重复文本的影响，额外去掉测试集中已在训练集出现的新闻，'
        '保留 1,137 篇，使用原模型重新计算指标。下表并列列出原测试集和过滤后测试集的分数。')
    d.table(['方法','原测试 Macro-F1','过滤后 Macro-F1'],[
        [display,pct(r[name]['test']['macro_f1']),
         pct(r[name]['duplicate_sensitivity']['test_without_exact_train_duplicates']['macro_f1'])]
        for name,display in zip(NAMES,DISPLAY)], [175,160,w-335])
    d.figure('score_intervals.png','2,000 次分层 bootstrap 得到的 Macro-F1 95% 区间')
    ci=stats['paired_bert_binary']['macro_f1_difference_bert_minus_binary_95ci']
    d.p('bootstrap 在三个类别内部有放回抽样，保持各类别的原测试样本数，随机种子为 123。'
        '各模型使用同一次抽样行号，按配对方式计算差值。模型保持固定，区间反映测试样本变化带来的分数波动。')
    d.p(f'BERT 减去 Binary 的 Macro-F1 差值区间为 [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] 个百分点。'
        '由于置信区间包含 0，目前不能认为 Binary BoW 一定优于 BERT。此外，BERT 的输入限制为 64 个 token，也可能影响了结果。')

    d.section('18 运行检查与实验总结')
    d.p('实验完成后，重新检查了表示计算、数据划分、任务配置和保存的预测。下面七项检查均通过。')
    test_text=(ROOT/'logs'/'report_test_check.log').read_text(encoding='utf-8').strip()
    d.log(test_text,'logs/report_test_check.log')
    d.h('运行与复现')
    d.log('python src/download_resources.py all\npython run_experiments.py --skip-download\n'
          'python src/report_evidence.py\npython src/build_report.py', '完整环境安装与解释器路径见 README.md')
    d.p('一键脚本顺序运行数据检查、六组训练、测试和报告生成。已经训练完成时，report_evidence.py 可以重新核算预测，'
        '不必再次训练。单项训练可以在 train_classical.py 中选择 bow、glove、w2v_ag 或 w2v_nyt；'
        '单篇预测使用 src/predict.py 和保存的模型。')
    d.h('本次实验得到的结论')
    d.p('这次实验中，最值得进一步分析的结果是 BERT 没有超过 Binary BoW。核对六组模型的测试行号和保存预测后，'
        '重新计算的指标与结果表一致。再查看输入长度，发现 99.91% 的测试新闻超过 64 个 token，BERT 实际上只读取了新闻开头，'
        '而词袋方法使用全文。这可能是本次两种方法表现不同的原因之一。')
    d.p('训练曲线也给出了一个具体观察：第三轮训练损失继续下降，验证 Macro-F1 却低于第二轮，'
        '因此最终采用第二轮模型。三组平均词向量中，NYT Word2Vec 表现最好，它的测试词项覆盖率也最高。'
        '后续可以尝试增加 BERT 输入长度，并用不同随机种子重复实验，观察这些结果是否稳定。')

    d.section('参考文献与代码索引')
    d.p('实验数据使用课程提供的 HW-1 文件夹。算法和工具文档如下；具体运行版本记录在结果文件中。')
    refs=[
        '[1] Pennington J, Socher R, Manning C D. GloVe: Global Vectors for Word Representation. 2014. https://nlp.stanford.edu/projects/glove/',
        '[2] Mikolov T, Chen K, Corrado G, Dean J. Efficient Estimation of Word Representations in Vector Space. 2013. https://arxiv.org/abs/1301.3781',
        '[3] Devlin J, Chang M W, Lee K, Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 2019. https://arxiv.org/abs/1810.04805',
        '[4] scikit-learn. Text feature extraction. https://scikit-learn.org/stable/modules/feature_extraction.html',
        '[5] Gensim Word2Vec API. https://radimrehurek.com/gensim/models/word2vec.html',
        '[6] BERT 模型资源. https://huggingface.co/google-bert/bert-base-uncased',
    ]
    for ref in refs: d.p(ref,'small')
    d.table(['代码文件','作用'],[
        ['src/common.py','读取与划分数据、统一分词、评价指标、保存预测。'],
        ['src/train_classical.py','两种词袋、三种词向量、Logistic Regression 选参。'],
        ['src/train_bert.py','编码数据、三轮微调、验证选模型、最终测试。'],
        ['src/predict.py','读取保存权重并预测单篇文本。'],
        ['src/report_evidence.py','核算六组测试预测，并重新执行正确性检查。'],
        ['src/report_charts.py','根据 JSON 结果生成九张数值图。'],
        ['src/report_log_images.py','将实际日志摘录排版为配图。'],
        ['src/build_report.py / report_details.py','整理报告正文，用 ReportLab 排版 PDF，并导出 Markdown。'],
        ['tests/test_correctness.py','七项语义与结果检查。'],
        ['run_experiments.py','完整实验入口。']], [185,w-185],padding=4.5)
    d.p(f"BERT 固定资源版本：{bert['metadata']['source']['revision']}。数据文件校验值见 data_audit.json，"
        '预训练权重与环境记录见 resource_provenance.json。逐篇预测保存在各模型 predictions.csv，'
        '代码和报告的文件校验值随 SUBMISSION_MANIFEST.json 提交。','small')
