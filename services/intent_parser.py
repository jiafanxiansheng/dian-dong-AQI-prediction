"""
用户意图解析器 — 从自然语言输入中识别地点和站点
"""
from config import SITE_DETAILS, SUPPORTED_LOCATIONS


def parse_user_intent(user_input: str) -> tuple[str, str, str]:
    """解析用户输入，提取地点/站点和意图

    Args:
        user_input: 用户原始输入文本

    Returns:
        tuple: (意图类型, 目标代码, 目标名称)
               意图类型: 'site' | 'location' | 'unsupported'
    """
    user_input_lower = user_input.lower()

    # 1) 精确匹配站点名称
    for site_code, site_info in SITE_DETAILS.items():
        site_name = site_info["name"]
        if site_name in user_input_lower or site_name in user_input:
            print(f"🎯 识别到具体站点：{site_name}（{site_code}）")
            return ("site", site_code, site_name)

    # 2) 匹配城市/地区名称
    for site_code, site_info in SITE_DETAILS.items():
        location = site_info["location"]
        if location in user_input_lower or location in user_input:
            print(f"🎯 识别到地区：{location}")
            return ("location", location, location)

    # 3) 检查不支持的地区关键词
    unsupported_keywords = [
        "昆明", "大理", "丽江", "玉溪", "楚雄", "红河",
        "西双版纳", "保山", "德宏", "怒江", "迪庆", "临沧",
        "普洱", "贵州", "四川",
        "北京", "上海", "广州", "深圳", "成都", "重庆",
        "贵阳", "南宁", "长沙", "武汉",
    ]

    for keyword in unsupported_keywords:
        if keyword in user_input:
            if keyword in SUPPORTED_LOCATIONS:
                print(f"🎯 识别到地区：{keyword}")
                return ("location", keyword, keyword)
            print(f"⚠️ 识别到不支持的地区：{keyword}")
            return ("unsupported", keyword, keyword)

    # 4) 默认回退到曲靖
    print("⚠️ 未识别到具体地点，默认查询曲靖地区")
    return ("location", "曲靖", "曲靖")
