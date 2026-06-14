"""
Preview label distribution for two new classification schemes.
Maps 500 open-labeled samples to each scheme using keyword rules.
"""

import json
from collections import Counter
from pathlib import Path


def load_open_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f
                if line.strip() and not json.loads(line.strip()).get("error")]


def classify_scheme1(label: str) -> str:
    """Scheme 1: 5 categories"""
    # 钓鱼邮件: account security, phishing, file lure
    if kw_match(label, ['验证', '钓鱼', '黑客', '密码', '盗号', '认证', '注销',
                        '锁定', '登录', '激活', '警报', '异常', '可疑',
                        '云存储', '共享文件', '共享','附件','系统通知','订单提醒',
                        '快递通知','邮件通知','邮箱','账户异常',
                        '安全警告','密码重置','账号安全','系统安全','系统警报']):
        return '钓鱼邮件'
    # 发票财务资产: invoice, accounting, financial
    if kw_match(label, ['发票', '税务', '票据', '增值税', '做账', '财务', '货款',
                        '询价', '虚假报告', '伪造报告', '报销', '抵扣', '采购',
                        '货款诈骗','采购诈骗','订单诈骗']):
        return '发票财务资产'
    # 色情暴力低俗: adult, violent, vulgar
    if kw_match(label, ['色情', '成人', '淫秽', '黄色', '情色', '偷拍',
                        '裸聊', '裸体', '壮阳', '童色', '性爱', '暴力',
                        '低俗', 'av', 'AV']):
        return '色情暴力低俗'
    # 论文会议推广: academic, conference, training
    if kw_match(label, ['学术会议', '学术征稿', '学术推广', '征稿', '论文', '期刊',
                        '培训', '讲座', '学位', '学术访问', '学术讲座', '研究',
                        '医学会议', '科技', '教育','留学','课程',
                        '会议通知','学术']):
        return '论文会议推广'
    # 商业博彩营销: commercial, gambling, dating, recruitment
    if kw_match(label, ['推广', '营销', '广告', '展会', '招标', '招商', '品牌',
                        '商务', '房产', '商业', '企业',
                        '赌博', '博彩', '赌场', '彩票', '投注', '网赌', '购彩',
                        '约炮', '约会', '交友', '社交', '婚恋', '裸聊',
                        '招聘', '直播','服务','资讯','合作']):
        return '商业博彩营销'
    # Fallbacks
    if kw_match(label, ['诈骗', '欺诈', '冒充', '虚假', '伪造', '勒索', '恐吓', '假冒', '骗']):
        return '发票财务资产'  # fraud → financial
    if kw_match(label, ['赚钱', '兼职', '网赚']):
        return '商业博彩营销'
    return None


def classify_scheme2(label: str) -> str:
    """Scheme 2: 5 categories"""
    # 色情暴力
    if kw_match(label, ['色情', '成人', '淫秽', '黄色', '情色', '偷拍',
                        '裸聊', '裸体', '壮阳', '童色', '性爱', '暴力',
                        '约炮', '交友', '学妹', '约会', '婚恋', '社交',
                        '援交', '一夜情', '炮友', 'av', 'AV']):
        return '色情暴力'
    # 推广营销
    if kw_match(label, ['推广', '营销', '广告', '展会', '招标', '招商', '品牌',
                        '商务', '房产', '商业', '企业', '学术', '会议', '征稿',
                        '论文', '期刊', '培训', '课程', '讲座', '学位', '教育',
                        '留学', '研究', '科技', '招聘', '直播', '资讯', '服务',
                        '合作', '医药','医疗','产品']):
        return '推广营销'
    # 钓鱼邮件
    if kw_match(label, ['验证', '钓鱼', '黑客', '密码', '盗号', '认证', '注销',
                        '锁定', '登录', '激活', '警报', '异常', '可疑',
                        '云存储', '共享文件', '共享', '附件', '系统通知', '订单提醒',
                        '快递通知', '邮件通知', '邮箱', '账户异常',
                        '安全警告', '密码重置', '系统安全', '系统警报']):
        return '钓鱼邮件'
    # 发票
    if kw_match(label, ['发票', '税务', '票据', '增值税']):
        return '发票'
    # 赌博/诈骗
    if kw_match(label, ['赌博', '博彩', '赌场', '彩票', '投注', '网赌', '购彩',
                        '诈骗', '欺诈', '冒充', '虚假', '伪造', '勒索', '恐吓',
                        '假冒', '骗', '货款', '询价', '采购', '虚假报告',
                        '伪造报告', '遗产', '贷款', '投资骗', '招聘骗', '赚钱',
                        '兼职', '网赚', '日赚']):
        return '赌博诈骗'
    return None


def kw_match(text: str, keywords: list) -> bool:
    return any(kw in text for kw in keywords)


def preview(scheme_name: str, classify_fn, labels_data: list) -> dict:
    total = len(labels_data)
    results = Counter()
    unmapped = []
    for r in labels_data:
        cat = classify_fn(r['llm_category'])
        if cat:
            results[cat] += 1
        else:
            results['未分类'] += 1
            unmapped.append(r['llm_category'])

    print(f"\n{'='*55}")
    print(f"  {scheme_name}")
    print(f"{'='*55}")
    print(f"  {'类别':14s}  {'数量':>5}  {'占比':>7s}")
    print(f"  {'─'*14}  {'─'*5}  {'─'*7}")
    vals = []
    for cat, cnt in results.most_common():
        pct = cnt / total * 100
        bar = '█' * max(1, int(pct / 2))
        print(f"  {cat:12s}  {cnt:>5}  {pct:>6.1f}%  {bar}")
        if cat != '未分类':
            vals.append(cnt)

    import statistics
    if len(vals) == 5:
        print(f"\n  极差: {max(vals)-min(vals)}  标准差: {statistics.stdev(vals):.0f}")

    if unmapped:
        from collections import Counter as C
        uc = C(unmapped)
        print(f"\n  未映射 ({len(unmapped)} 条, {len(uc)} 个标签):")
        for kw, c in uc.most_common(15):
            print(f"    {kw}: {c}")

    return dict(results)


def main():
    path = Path("data/open_sample_500_labeled.jsonl")
    labels_data = load_open_labels(str(path))
    print(f"Loaded {len(labels_data)} open-labeled samples")

    # Scheme 1
    r1 = preview("方案1: 钓鱼邮件 / 发票财务资产 / 色情暴力低俗 / 论文会议推广 / 商业博彩营销",
                  classify_scheme1, labels_data)

    # Scheme 2
    r2 = preview("方案2: 色情暴力 / 推广营销 / 钓鱼邮件 / 发票 / 赌博诈骗",
                  classify_scheme2, labels_data)

    # Comparison
    print(f"\n\n{'='*55}")
    print(f"  两方案并排对比")
    print(f"{'='*55}")
    cats1 = ['钓鱼邮件', '发票财务资产', '色情暴力低俗', '论文会议推广', '商业博彩营销']
    cats2 = ['色情暴力', '推广营销', '钓鱼邮件', '发票', '赌博诈骗']
    print(f"  {'方案1':18s}  {'%':>6s}  │  {'方案2':18s}  {'%':>6s}")
    print(f"  {'─'*18}  {'─'*6}  ──  {'─'*18}  {'─'*6}")
    for c1, c2 in zip(cats1, cats2):
        p1 = r1.get(c1, 0) / len(labels_data) * 100
        p2 = r2.get(c2, 0) / len(labels_data) * 100
        print(f"  {c1:16s}  {p1:>5.1f}%  │  {c2:16s}  {p2:>5.1f}%")

    # Show top labels in each category
    print(f"\n\n{'='*55}")
    print(f"  各方案每类包含的主要open标签")
    print(f"{'='*55}")
    for scheme_name, classify_fn in [("方案1", classify_scheme1), ("方案2", classify_scheme2)]:
        print(f"\n--- {scheme_name} ---")
        cat_labels = {}
        for r in labels_data:
            cat = classify_fn(r['llm_category'])
            kw = r['llm_category']
            if cat:
                if cat not in cat_labels:
                    cat_labels[cat] = Counter()
                cat_labels[cat][kw] += 1
        for cat, cnts in sorted(cat_labels.items()):
            total_c = sum(cnts.values())
            top = cnts.most_common(8)
            print(f"\n  [{total_c}封] {cat}:")
            print(f"    {', '.join(f'{k}({v})' for k,v in top)}")


if __name__ == "__main__":
    main()
