"""
Cluster open-mode LLM labels into 5 balanced categories.
Reads open_sample_500_labeled.jsonl and proposes category groupings.
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

    print(f"Total: {total}")
    print()

    # Top labels
    print("Top 40 labels:")
    for label, c in cnt.most_common(40):
        print(f"  {label}  {c}  ({c/total*100:.1f}%)")
    print(f"  ... {len(cnt)-40} more labels: {sum(v for _,v in cnt.most_common()[40:])}")
    print()

    # ==========================================
    # Manual mapping — every label assigned
    # ==========================================
    cat_map = {}

    # Category 1: 色情推广 (explicit adult — porn, adult content)
    c1 = [
        "色情推广", "色情博文", "色情资源下载", "成人内容", "色情直播",
        "色情网", "童色情", "黄色直播", "成人视频", "色情诈骗",
        "成人保健品推广", "成人壮阳", "成人交", "色情约炮",
    ]
    for kw in c1:
        cat_map[kw] = "1_色情推广"

    # Category 2: 假发票与诈骗 (fake invoices, tax fraud, all financial scams)
    c2 = [
        "假发票", "税务发票", "发票代办", "发票", "发票办理",
        "虚假报告", "伪造报告", "询价诈骗", "货款询价", "购物诈骗",
        "电信诈骗", "诈骗邮件", "冒充诈骗", "虚构诈骗", "快递诈骗",
        "遗产诈骗", "冒充公检", "宗教诈骗", "购彩诈骗", "仿冒诈骗",
        "勒索诈骗", "虚构恐吓", "投资诈骗", "招聘诈骗",
        "货款诈骗", "采购诈骗", "订单诈骗", "发票办理",
        "冒充诈骗", "假冒诈骗", "诈骗推广",
    ]
    for kw in c2:
        cat_map[kw] = "2_假发票诈骗"

    # Category 3: 商业学术推广 (marketing, conferences, academic, ads)
    c3 = [
        "营销推广", "学术征稿", "培训推广", "学术会议推广", "论文润色",
        "学术会议", "产品推广", "软件推广", "商业推广", "期刊征稿",
        "学术推广", "企业推广", "展会推广", "设备推销", "招标采购",
        "招标信息", "招标推广", "医学会议", "品牌推广", "网站推广",
        "教育推广", "企业服务推广", "签证服务", "技术服务",
        "企业招聘", "外教招聘", "投资推广", "科技推广", "商业地产",
        "商品推广", "低促", "广告营销", "课程推广", "房产推广",
        "招商推广", "商业服务推广", "期刊推广", "学术讲座", "体育推广",
        "房产资讯", "医疗设备推广", "金融服务", "企业询价",
        "企业资讯", "企业商业推广", "商业资讯", "展会信息",
        "广告推广", "招聘推广", "活动推广", "服务推广", "旅游推广",
        "农业推广", "建材推广", "科技资讯", "学术访问", "学术会议通知",
        "学术学位认证", "学术讲座", "培训课程", "教育资讯",
        "留学推广", "会议通知", "商务合作", "商业合作", "合作推广",
        "医药推广", "医药广告", "药品推广", "壮阳药推广",
        "直播推广",
    ]
    for kw in c3:
        cat_map[kw] = "3_商业推广"

    # Category 4: 账号钓鱼与博彩 (phishing + gambling)
    c4 = [
        "账号验证", "账户验证", "邮件钓鱼", "快递钓鱼", "文件钓鱼",
        "云存储钓鱼", "订单钓鱼", "密码钓鱼", "钓鱼网站", "黑客入侵",
        "系统警报", "钓鱼恐吓", "密码通知", "账号安全", "账号认证",
        "钓鱼验证", "邮箱钓鱼", "文件共享钓鱼", "系统通知", "订单提醒",
        "邮件通知", "系统安全", "安全警告", "密码重置", "快递通知",
        "邮箱异常", "账户异常", "可疑附件", "云存储诈骗",
        "钓鱼诈骗", "钓鱼", "账号盗取", "钓鱼验证",
        "赌博推广", "赌博诈骗", "网投推广", "快速赚钱",
        "购彩", "彩票", "博彩", "赌场", "投注", "赌博", "网赌",
        "账户验证", "系统警报", "系统安全",
    ]
    for kw in c4:
        cat_map[kw] = "4_钓鱼博彩"

    # Category 5: 约会交友 (dating, social encounters)
    c5 = [
        "色情交友", "学妹约炮", "约炮诈骗", "社交推广", "婚恋交友",
        "约会诈骗", "裸聊诈骗", "裸聊推广", "交友推广",
        "学妹约炮", "约会", "裸聊", "色情交友",
        "社交推广", "婚恋","交友","约炮","约会推广",
    ]
    for kw in c5:
        cat_map[kw] = "5_约会交友"

    # Some remaining labels — assign by keyword matching
    remaining = [kw for kw in cnt if kw not in cat_map]
    for kw in remaining:
        if any(w in kw for w in ["色情","成人","黄色","裸","淫秽","情色","性"]):
            cat_map[kw] = "1_色情推广"
        elif any(w in kw for w in ["发票","税务","票据","诈骗","欺诈","冒充","虚假","伪造","勒索","贷款"]):
            cat_map[kw] = "2_假发票诈骗"
        elif any(w in kw for w in ["推广","营销","广告","培训","会议","征稿","论文","期刊","展会","展览","招标","招商","课程","品牌","房产","医疗","教育","留学","签证","讲座","商务","合作"]):
            cat_map[kw] = "3_商业推广"
        elif any(w in kw for w in ["验证","钓鱼","账号","账户","密码","安全","登录","黑客","通知","提醒","警报","附件","分享","云","存储","盗号"]):
            cat_map[kw] = "4_钓鱼博彩"
        elif any(w in kw for w in ["赌博","博彩","赌场","彩票","赌球","投注","赚钱","网赚","兼职","约炮","约","交友","社交","婚恋","裸聊","直播"]):
            cat_map[kw] = "5_约会交友"
        else:
            cat_map[kw] = "4_钓鱼博彩"  # default: misc → phishing/gambling bucket

    # Count
    results = Counter()
    unmapped = []
    for l in labels:
        cat = cat_map.get(l["llm_category"], "")
        if cat:
            results[cat] += 1
        else:
            unmapped.append(l["llm_category"])
            results["未映射"] += 1

    short = {
        "1_色情推广": "色情推广",
        "2_假发票诈骗": "假发票诈骗",
        "3_商业推广": "商业推广",
        "4_钓鱼博彩": "钓鱼博彩",
        "5_约会交友": "约会交友",
    }

    print("=" * 55)
    print("  新 5 类分布 (基于 500 条 open 标注)")
    print("=" * 55)
    vals = []
    for cid in ["1_色情推广", "2_假发票诈骗", "3_商业推广", "4_钓鱼博彩", "5_约会交友"]:
        c = results.get(cid, 0)
        pct = c / total * 100
        bar = "█" * max(1, int(pct))
        print(f"  {short[cid]:10s}  {c:>4}  ({pct:5.1f}%)  {bar}")
        vals.append(c)
    um = results.get("未映射", 0)
    if um:
        print(f"  {'未映射':10s}  {um:>4}")

    import statistics

    print(f"\n极差: {max(vals)-min(vals)}  标准差: {statistics.stdev(vals):.0f}")
    print()

    # Comparison
    print("=== 新旧 5 类对比 ===")
    old_pct = {"色情": 33.2, "发票": 21.9, "营销": 23.8, "钓鱼": 18.0, "赌博": 3.1}
    new_map = {"色情": "1_色情推广", "发票": "2_假发票诈骗", "营销": "3_商业推广",
               "钓鱼": "4_钓鱼博彩", "赌博": "5_约会交友"}
    print(f"  {'':12s}  {'旧':>6s}  {'新':>6s}  {'变化':>8s}")
    for old_name, new_id in new_map.items():
        o = old_pct[old_name]
        n = results.get(new_id, 0) / total * 100
        d = n - o
        sign = "+" if d > 0 else ""
        print(f"  {short[new_id]:10s}  {o:>5.1f}%  {n:>5.1f}%  {sign}{d:>+7.1f}%")

    # Label mapping for retrain
    print(f"\n=== 类别定义 (用于后续训练) ===")
    for cid in ["1_色情推广", "2_假发票诈骗", "3_商业推广", "4_钓鱼博彩", "5_约会交友"]:
        print(f"\n{short[cid]}:")
        # Show the open labels mapped to this category
        mapped_labels = {kw for kw, c in cat_map.items() if c == cid}
        sub_count = sum(cnt.get(kw, 0) for kw in mapped_labels)
        top = sorted(mapped_labels, key=lambda k: cnt.get(k, 0), reverse=True)[:10]
        print(f"  样本数: {sub_count}")
        print(f"  主要标签: {', '.join(top)}")


if __name__ == "__main__":
    main()
