"""
Design 5 BALANCED categories by clustering 103 open LLM labels.
Goal: each category ~20% of data, semantically coherent.
"""

import json
from collections import Counter
from pathlib import Path


def main():
    path = Path("data/open_sample_500_labeled.jsonl")
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line.strip())
            if r and not r.get("error"):
                labels.append(r)

    total = len(labels)
    cnt = Counter(l["llm_category"] for l in labels)

    # ==========================================
    # 5 balanced categories — manual semantic clustering
    # Assign every label to exactly one category
    # Target: ~100 each (20% of 500)
    # ==========================================

    cat_map = {}

    # ---- Category 1: 色情内容推广 (~100) ----
    # Take the core 色情推广 but leave some for other categories
    c1 = [
        "色情推广",       # 131 — too big! split: ~100 here, rest to other
        "色情博文",       # 2
        "色情资源下载",   # 1
        "成人内容",       # merged below
        "成人保健品推广", # merged below
    ]
    # Actually 色情推广 is 131 alone. We CAN'T split it arbitrarily.
    # Instead, put ALL of 色情推广 into one category and balance others.

    # Let me try a completely different approach:
    # Group by SEMANTIC proximity, then adjust by moving smaller groups

    cat_map.clear()

    # === Category A: 色情淫秽 (all adult/porn content) ===
    # 色情推广(131) + 色情博文(2) + 色情资源下载(1) + 成人 content + 色情直播 etc.
    cat_a_labels = [
        "色情推广",         # 131
        "色情博文",         # 2
        "色情资源下载",     # 1
        "成人内容",         # (merged)
        "色情直播",         # (merged)
        "色情网",           # (merged)
        "童色情",           # (merged)
        "黄色直播",         # (merged)
        "成人视频",         # (merged)
        "色情诈骗",         # 1
        "成人保健品推广",   # (merged)
        "成人壮阳",         # (merged)
        "成人交",           # (merged)
    ]
    cat_a = sum(cnt.get(kw, 0) for kw in cat_a_labels)

    # === Category B: 假发票与财务诈骗 (~105) ===
    cat_b_labels = [
        "假发票",           # 102
        "税务发票",         # 9
        "发票代办",         # (merged)
        "发票",             # (merged)
        "发票办理",         # (merged)
        "发票办理",         # duplicate
        "货款询价",         # (merged)
        "询价诈骗",         # 2
        "购物诈骗",         # (merged)
        "虚假报告",         # (merged)
        "伪造报告",         # (merged)
        "货款诈骗",         # (merged)
        "采购诈骗",         # (merged)
        "订单诈骗",         # (merged)
    ]
    cat_b = sum(cnt.get(kw, 0) for kw in cat_b_labels)

    # === Category C: 商业营销推广 (~80-100) ===
    cat_c_labels = [
        "营销推广",         # 20
        "学术征稿",         # 10
        "学术会议推广",     # 7
        "论文润色",         # 5
        "学术会议",         # 4
        "产品推广",         # 4
        "培训推广",         # 3
        "软件推广",         # 3
        "商业推广",         # 2
        "期刊征稿",         # 2
        "学术推广",         # 2
        "企业推广",         # 2
        "招标采购",         # 2
        "展会推广",         # 1
        "设备推销",         # 1
        "招标信息",         # 1
        "招标推广",         # 1
        "医学会议",         # 1
        "品牌推广",         # 1
        "网站推广",         # 1
        "教育推广",         # 1
        "企业服务推广",     # (merged)
        "签证服务",         # (merged)
        "技术服务",         # (merged)
        "企业招聘",         # (merged)
        "外教招聘",         # (merged)
        "投资推广",         # (merged)
        "科技推广",         # (merged)
        "商业地产",         # (merged)
        "商品推广",         # (merged)
        "低促",             # (merged)
        "广告营销",         # (merged)
        "课程推广",         # (merged)
        "房产推广",         # (merged)
        "招商推广",         # (merged)
        "商业服务推广",     # (merged)
        "期刊推广",         # (merged)
        "学术讲座",         # (merged)
        "体育推广",         # (merged)
        "房产资讯",         # (merged)
        "医疗设备推广",     # (merged)
        "金融服务",         # (merged)
        "企业询价",         # (merged)
        "企业资讯",         # (merged)
        "企业商业推广",     # 1
        "商业资讯",         # (merged)
        "广告推广",         # (merged)
        "旅游推广",         # (merged)
        "农业推广",         # (merged)
        "建材推广",         # (merged)
        "医药推广",         # (merged)
        "医药广告",         # (merged)
        "药品推广",         # 4
        "壮阳药推广",       # (merged)
        "直播推广",         # 1
        "学术会议通知",     # (merged)
        "学术学位认证",     # (merged)
        "学术访问",         # (merged)
        "培训课程",         # (merged)
        "教育资讯",         # (merged)
        "留学推广",         # (merged)
        "会议通知",         # (merged)
        "商务合作",         # (merged)
        "商业合作",         # (merged)
        "合作推广",         # (merged)
        "科技资讯",         # (merged)
    ]
    cat_c = sum(cnt.get(kw, 0) for kw in cat_c_labels)

    # === Category D: 钓鱼诈骗 (~80-100) ===
    cat_d_labels = [
        "账号验证",         # 21
        "账户验证",         # 12
        "邮件钓鱼",         # 6
        "快递钓鱼",         # 3
        "文件钓鱼",         # 3
        "云存储钓鱼",       # 3
        "订单钓鱼",         # 3
        "密码钓鱼",         # 3
        "钓鱼网站",         # 2
        "黑客入侵",         # 1
        "系统警报",         # 1
        "钓鱼恐吓",         # (merged)
        "密码通知",         # 1
        "账号安全",         # (merged)
        "账号认证",         # 1
        "钓鱼验证",         # (merged)
        "邮箱钓鱼",         # (merged)
        "文件共享钓鱼",     # (merged)
        "系统通知",         # 3
        "订单提醒",         # (merged)
        "邮件通知",         # 1
        "系统安全",         # (merged)
        "安全警告",         # (merged)
        "密码重置",         # (merged)
        "快递通知",         # (merged)
        "邮箱异常",         # (merged)
        "账户异常",         # (merged)
        "可疑附件",         # (merged)
        "云存储诈骗",       # (merged)
        "钓鱼诈骗",         # (merged)
        "钓鱼",             # (merged)
        "账号盗取",         # (merged)
        "电信诈骗",         # 2
        "诈骗邮件",         # 1
        "冒充诈骗",         # 1
        "虚构诈骗",         # (merged)
        "快递诈骗",         # (merged)
        "遗产诈骗",         # (merged)
        "冒充公检",         # (merged)
        "宗教诈骗",         # (merged)
        "购彩诈骗",         # (merged)
        "仿冒诈骗",         # (merged)
        "勒索诈骗",         # (merged)
        "虚构恐吓",         # (merged)
        "投资诈骗",         # (merged)
        "招聘诈骗",         # (merged)
    ]
    cat_d = sum(cnt.get(kw, 0) for kw in cat_d_labels)

    # === Category E: 约会社交与赌博博彩 (~70-90) ===
    cat_e_labels = [
        "色情约炮",         # 30
        "赌博推广",         # 14
        "色情交友",         # 4
        "学妹约炮",         # 3
        "约炮诈骗",         # 2
        "社交推广",         # (merged)
        "婚恋交友",         # (merged)
        "约会诈骗",         # (merged)
        "裸聊诈骗",         # 1
        "裸聊推广",         # (merged)
        "交友推广",         # (merged)
        "约会",             # (merged)
        "裸聊",             # (merged)
        "赌博诈骗",         # 1
        "网投推广",         # (merged)
        "快速赚钱",         # 1
        "购彩",             # (merged)
        "彩票",             # (merged)
        "博彩",             # (merged)
        "赌场",             # (merged)
        "投注",             # (merged)
        "赌博",             # (merged)
        "网赌",             # (merged)
    ]
    cat_e = sum(cnt.get(kw, 0) for kw in cat_e_labels)

    # Collect all assigned labels
    assigned = set(cat_a_labels + cat_b_labels + cat_c_labels + cat_d_labels + cat_e_labels)

    # Show results
    names = {0: "A_色情内容", 1: "B_假发票诈骗", 2: "C_商业推广",
             3: "D_钓鱼诈骗", 4: "E_约会博彩"}
    counts = [cat_a, cat_b, cat_c, cat_d, cat_e]

    print(f"基于 500 条 open 标注的 5 类均衡方案\n")
    print(f"{'类别':16s}  {'数量':>5}  {'占比':>7s}  {'分布'}")
    print("-" * 55)
    for i in range(5):
        c = counts[i]
        pct = c / total * 100
        bar = "█" * max(1, int(pct))
        print(f"  {names[i]:14s}  {c:>5}  {pct:>6.1f}%  {bar}")

    import statistics
    print(f"\n极差: {max(counts)-min(counts)}  标准差: {statistics.stdev(counts):.0f}")
    print(f"目标: 每类 ~{total//5} (20%)")

    # Remaining unassigned
    remaining = {kw: c for kw, c in cnt.items() if kw not in assigned}
    rem_total = sum(remaining.values())
    print(f"\n未分配: {rem_total} 条 ({len(remaining)} 个标签)")

    # Print remaining labels for manual assignment
    if remaining:
        print("\n未分配标签 (需手动归入某类):")
        for kw, c in sorted(remaining.items(), key=lambda x: -x[1]):
            print(f"  {kw:20s}  {c}")

    # Suggest assignments for remaining
    print("\n\n=== 建议: 类定义 ===")
    definitions = {
        "A_色情淫秽": "色情网站、AV视频、成人内容、色情资源推广。不包括约会交友。",
        "B_假发票诈骗": "代开发票、增值税发票、货款询价、虚假报告等财务诈骗。",
        "C_商业推广": "展会、招标、广告营销、会议征稿、学术期刊、培训课程等商业/学术推广。",
        "D_钓鱼诈骗": "账号验证、密码窃取、虚假安全通知、快递钓鱼、系统通知欺诈。",
        "E_约会博彩": "约炮交友、线上赌场、体育博彩、彩票投注、快速赚钱等社交博彩类。",
    }
    for k, v in definitions.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
