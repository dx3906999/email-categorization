# 垃圾邮件分类系统 (v3)

基于 BERT 的中文垃圾邮件多分类系统。将 24,000 封已拦截的垃圾邮件分为 5 个子类别。

> **v3 更新：** 重构 5 类分类体系。最小类从 3.1% 提升到 13.3%，极差从 30.1 降到 20.2。

## 项目结构

```
├── src/
│   ├── parser.py                  # 日志解析器 (TSV+JSON → EmailRecord)
│   ├── db.py                      # SQLite 存储 (FTS5 全文索引 + 分类标签)
│   ├── classifier.py              # 分类器 (规则 + BERT融合 + IP传播)
│   ├── llm_label.py               # LLM 标注脚本 (DeepSeek/OpenAI/Anthropic)
│   ├── bert_dataset.py            # 数据集准备 (JSONL → HuggingFace Dataset)
│   ├── bert_train.py              # BERT 训练 + 评估 + 推理
│   ├── make_datasets.py           # 采样工具 (验证集/训练集/保留集)
│   ├── compare_strategies.py      # 三策略全量对比 (规则/BERT/融合)
│   ├── compare_low_confidence.py  # 低置信度区间分析
│   ├── validate_sample.py         # LLM 抽样验证
│   ├── export_labels.py           # 导出 record_id + spam_type
│   ├── cluster_labels.py          # 标签聚类 (凝聚聚类)
│   ├── cluster_traditional.py     # 传统聚类 (embedding + k-means)
│   ├── design_balanced_cats.py    # 均衡类别设计
│   ├── open_label_cluster.py      # Open 标签分析
│   ├── preview_labels.py          # 分类方案分布预览
│   ├── main.py                    # 入口
│   └── rules/                     # 规则文件 (JSON, 共 72 条)
│       ├── adult.json             # 色情 — 20 条
│       ├── gambling.json          # 赌博 — 9 条
│       ├── marketing.json         # 营销 — 13 条
│       ├── phishing.json          # 钓鱼 — 13 条
│       └── fraud.json             # 诈骗 — 17 条
├── data/
│   ├── scheme1_sample.jsonl                  # 方案1 训练样本 (2,400条)
│   ├── scheme1_sample_labeled.jsonl          # LLM 标注结果
│   ├── open_sample_500.jsonl                 # Open模式 探索样本 (500条)
│   ├── open_sample_500_labeled.jsonl         # Open标注结果
│   ├── holdout.jsonl                         # 保留集 (21,548条)
│   ├── train.jsonl                           # 旧训练集 (2,080条)
│   ├── quality_verification.jsonl            # 旧验证集 (372条)
│   ├── email_labels.jsonl                    # 最终导出 (24,000条)
│   ├── email_labels.csv                      # 最终导出 (CSV)
│   └── processed/                            # BERT 训练数据
├── models/
│   └── bert_classifier/                      # 训练好的 BERT 模型
├── database/
│   └── spam_email_data.log                   # 原始日志 (24,000条)
├── report.md                                 # 实验报告 (v3)
├── requirements.txt
└── .env                                      # API Key (不提交)
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 3. 导入日志到数据库
python src/main.py pipeline

# 4. 抽样 + LLM 标注 (方案1类别)
python src/llm_label.py data/scheme1_sample.jsonl --concurrency 10

# 5. 训练 BERT
python src/bert_dataset.py
python src/bert_train.py --train

# 6. 全量分类 + 导出
python src/compare_strategies.py --output data/strategy_comparison_scheme1.json
python src/export_labels.py --source bert --input data/strategy_comparison_scheme1.json
```

## 分类类别 (v3)

| 类别 | 标识 | 数字 | 说明 | 占比 |
|------|------|:---:|------|:----:|
| 色情暴力 | `adult_violence` | 0 | 色情网站、成人内容、AV视频、偷拍、裸聊约炮、暴力低俗 | 33.5% |
| 博彩营销 | `commercial` | 1 | 产品推广、展会搭建、招标采购、招聘、博彩、体育赌博、彩票 | 13.3% |
| 钓鱼邮件 | `phishing` | 2 | 账号验证、邮箱验证、密码窃取、虚假通知、文件诱导、快递诈骗 | 15.6% |
| 发票财务 | `finance` | 3 | 代开发票、增值税发票、做账报销、财务询价、货款诈骗 | 21.9% |
| 学术推广 | `academic` | 4 | 学术会议征稿、论文润色、期刊投稿、培训课程、学位认证 | 15.7% |

## 技术路线

```
原始日志 → LLM open探索(500条) → 设计5类体系 → LLM标注(2,400条) → BERT训练 → 纯BERT分类 → 导出
```

### BERT 输入格式

```
{from_name} [SEP] {subject} [SEP] {content}
```

## 核心命令

### 日志导入与查询

```bash
python src/db.py import                         # 导入日志
python src/db.py report                         # 统计报告
python src/db.py categories                     # 分类分布
python src/db.py search "phishing"              # 全文搜索
python src/db.py top sender                     # 发件人排名
```

### 分类

```bash
python src/classifier.py classify --use-bert                    # BERT 全量分类
python src/classifier.py predict "主题" "内容" --use-bert       # 单条预测
python src/classifier.py predict --record-id <id> --use-bert    # 从数据库查
```

### 训练流程

```bash
# 探索类别分布
python src/preview_labels.py                     # 预览方案分布
python src/llm_label.py data/open_sample.jsonl --mode open

# 标注训练集
python src/llm_label.py data/scheme1_sample.jsonl --concurrency 10

# 训练
python src/bert_dataset.py
python src/bert_train.py --train

# 全量分类 + 导出
python src/compare_strategies.py --output data/strategy_comparison_scheme1.json
python src/export_labels.py --source bert
```

### 分析工具

```bash
python src/compare_strategies.py                              # 三策略全量对比
python src/compare_low_confidence.py --extract --compare      # 低置信度分析
python src/validate_sample.py --sample-only                   # LLM验证抽样
python src/export_labels.py --source bert --format csv        # 导出
python src/llm_label.py data/xxx.jsonl --dry-run              # 查看prompt
```

## 核心指标 (v3)

| 指标 | 数值 |
|------|:----:|
| BERT 测试集 macro-F1 | 0.916 |
| BERT 测试集 accuracy | 93.0% |
| 全量覆盖率 | 100% |
| 置信度 ≥0.9 占比 | 95.2% |
| 最小类占比 | 13.3%（v1/v2 仅 3.1%） |
| 极差 | 20.2（v1/v2 为 30.1） |
| LLM 蒸馏总成本 | ~¥1.29 |

> v3 结论：重构 5 类体系后，所有类别占比均 > 13%，分布更加均衡。详见 [report.md](report.md)。

## 依赖

```
torch>=2.0
transformers>=4.40
datasets>=2.18
scikit-learn>=1.4
accelerate>=0.28
sentence-transformers>=3.0
httpx>=0.27
python-dotenv>=1.0
```
