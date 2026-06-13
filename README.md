# 垃圾邮件分类系统 (v2)

基于 BERT 的中文垃圾邮件多分类系统。将 24,000 封已拦截的垃圾邮件分为 5 个子类别：色情、赌博、营销、钓鱼、诈骗。

> **v2 更新：** 加入 `from_name` 特征，LLM 验证 Score 89.28 → 90.80。结论逆转：规则不再有益，推荐纯 BERT。

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
│   ├── main.py                    # 入口
│   └── rules/                     # 规则文件 (JSON, 共 72 条)
│       ├── adult.json             # 色情 — 20 条
│       ├── gambling.json          # 赌博 — 9 条
│       ├── marketing.json         # 营销 — 13 条
│       ├── phishing.json          # 钓鱼 — 13 条
│       └── fraud.json             # 诈骗 — 17 条
├── data/
│   ├── quality_verification.jsonl           # 验证集 (372条)
│   ├── quality_verification_labeled.jsonl   # LLM 验证标注
│   ├── train.jsonl                          # 训练集 (2,080条)
│   ├── train_labeled.jsonl                  # LLM 训练标注
│   ├── holdout.jsonl                        # 保留集 (21,548条)
│   ├── validation_sample.jsonl              # 随机验证样本 (1,000条)
│   ├── validation_sample_labeled.jsonl      # LLM 验证标注
│   ├── low_confidence_sample.jsonl          # 低置信度样本 (404条)
│   ├── low_confidence_sample_labeled.jsonl  # LLM 标注
│   ├── strategy_comparison.json             # 三策略全量对比结果
│   ├── email_labels.jsonl                   # 最终导出 (24,000条)
│   ├── email_labels.csv                     # 最终导出 (CSV)
│   └── processed/                           # BERT 训练数据
├── models/
│   └── bert_classifier/                     # 训练好的 BERT 模型
├── database/
│   └── spam_email_data.log                  # 原始日志 (24,000条)
├── report.md                                # 实验报告 (v2)
├── requirements.txt
└── .env                                     # API Key (不提交)
```

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY

# 4. 导入日志到数据库 + 规则初筛
python src/main.py pipeline

# 5. 生成数据集 (含 from_name 特征)
python src/make_datasets.py

# 6. LLM 标注
python src/llm_label.py data/quality_verification.jsonl --concurrency 10
python src/llm_label.py data/train.jsonl --concurrency 10

# 7. 训练 BERT
python src/bert_dataset.py
python src/bert_train.py --train

# 8. 全量分类 + 导出
python src/compare_strategies.py                    # 三策略对比
python src/export_labels.py --source bert           # 导出纯BERT结果
```

## 分类类别

| 类别 | 标识 | 数字 | 说明 | 占比 |
|------|------|:---:|------|:----:|
| 色情淫秽 | `adult` | 0 | 成人内容、色情网站、约炮交友、AV 视频 | 33.2% |
| 假发票/诈骗 | `fraud` | 1 | 代开发票、增值税、中奖诈骗、遗产诈骗 | 21.9% |
| 赌博博彩 | `gambling` | 2 | 线上赌场、体育博彩、彩票投注 | 3.1% |
| 营销推广 | `marketing` | 3 | 展会搭建、招标采购、广告营销、会议征稿 | 23.8% |
| 钓鱼诈骗 | `phishing` | 4 | 账号验证、密码窃取、虚假通知、快递通知 | 18.0% |

## 技术路线

```
原始日志 → 规则引擎(初筛) → LLM标注(含from_name) → BERT训练 → 纯BERT分类 → 导出
```

### BERT 输入格式

```
{from_name} [SEP] {subject} [SEP] {content}
```

## 核心命令

### 日志导入与查询

```bash
python src/db.py import                     # 导入日志
python src/db.py report                     # 统计报告
python src/db.py categories                 # 分类分布
python src/db.py search "phishing"          # 全文搜索
python src/db.py top sender                 # 发件人排名
```

### 分类

```bash
# BERT 分类（推荐）
python src/classifier.py classify --use-bert                    # 全量融合
python src/classifier.py predict "主题" "内容" --use-bert       # 单条预测
python src/classifier.py predict --record-id <id> --use-bert    # 从数据库查

# 规则分类（仅作参考）
python src/classifier.py classify --no-ip                       # 仅规则
python src/classifier.py rules                                  # 查看规则
```

### BERT 训练

```bash
python src/llm_label.py data/train.jsonl --concurrency 10    # LLM 标注训练集
python src/bert_dataset.py                                    # 准备训练数据
python src/bert_train.py --train                              # 训练
python src/bert_train.py --eval-only                          # 评估
```

### 评估与导出

```bash
python src/compare_strategies.py                             # 三策略全量对比
python src/compare_low_confidence.py --extract               # 提取低置信度样本
python src/compare_low_confidence.py --compare               # 低置信度策略对比
python src/validate_sample.py --sample-only                  # 创建LLM验证样本
python src/export_labels.py --source bert                    # 导出纯BERT结果
python src/export_labels.py --source bert --format csv       # 仅CSV
```

### 数据集管理

```bash
python src/make_datasets.py                                  # 生成采样
python src/llm_label.py data/xxx.jsonl --dry-run             # 查看 prompt
```

## 核心指标 (v2)

| 指标 | 数值 |
|------|:----:|
| BERT 测试集 macro-F1 | 0.944 |
| LLM 验证准确率 (996条) | 95.6% |
| 综合评分 Mean IoU | **90.80** |
| 低置信区间 BERT 准确率 | 53.8% |
| 低置信区间混合策略效果 | -5.1%（有害） |
| LLM 蒸馏总成本 | ~¥1.64 |

> v2 结论：纯 BERT 在所有场景下均优于混合策略，规则引擎无补充价值。详见 [report.md](report.md)。

## 依赖

```
torch>=2.0
transformers>=4.40
datasets>=2.18
scikit-learn>=1.4
accelerate>=0.28
httpx>=0.27
python-dotenv>=1.0
```
