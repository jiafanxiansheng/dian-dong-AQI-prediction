"""
统一配置中心 — 滇东AQI智能预测系统
所有配置项优先从环境变量读取，未设置则使用默认值。
"""
import os

# ─── 项目根目录 ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 数据库配置 ───────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "database": os.getenv("DB_NAME", "air_data"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
}

# ─── DeepSeek API 配置 ────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
)

# ─── 模型目录 ─────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "模型")
COMPARISON_DIR = os.path.join(BASE_DIR, "模型对比")

# ─── 预测参数 ─────────────────────────────────────────────
FORECAST_HOURS_LIST = [6, 12]
PRIMARY_FORECAST_HOURS = 12
LAG_HOURS = 23

# ─── 站点详情（8个监测站） ────────────────────────────────
SITE_DETAILS = {
    "1916A": {"name": "环境监测站", "location": "曲靖"},
    "1917A": {"name": "烟厂办公区", "location": "曲靖"},
    "2596A": {"name": "监测站", "location": "昭通"},
    "2610A": {"name": "州水务局", "location": "文山州"},
    "2611A": {"name": "市便民服务中心", "location": "文山州"},
    "3376A": {"name": "南苑二区", "location": "曲靖"},
    "3377A": {"name": "曲靖师范学院", "location": "曲靖"},
}

# ─── 地区到城市的映射（用于天气查询） ────────────────────
LOCATION_CITY_MAP = {
    "曲靖": "曲靖",
    "昭通": "昭通",
    "文山州": "文山",
}

SUPPORTED_LOCATIONS = set(info["location"] for info in SITE_DETAILS.values())

# ─── 天气英文→中文翻译映射 ──────────────────────────────
WEATHER_ZH_MAP = {
    "Sunny": "晴",
    "Clear": "晴",
    "Partly cloudy": "多云",
    "Partly Cloudy": "多云",
    "Cloudy": "阴",
    "Overcast": "阴",
    "Mist": "薄雾",
    "Fog": "雾",
    "Freezing fog": "冻雾",
    "Patchy rain nearby": "局部小雨",
    "Patchy rain possible": "可能有零星小雨",
    "Light rain": "小雨",
    "Light rain shower": "小阵雨",
    "Moderate rain": "中雨",
    "Moderate rain at times": "时有中雨",
    "Heavy rain": "大雨",
    "Heavy rain at times": "时有大雨",
    "Light drizzle": "小毛毛雨",
    "Patchy light drizzle": "局部小毛毛雨",
    "Torrential rain": "暴雨",
    "Patchy snow nearby": "局部小雪",
    "Light snow": "小雪",
    "Moderate snow": "中雪",
    "Heavy snow": "大雪",
    "Patchy light rain": "局部小雨",
    "Thundery outbreaks possible": "可能有雷阵雨",
    "Thundery outbreaks in nearby": "附近有雷阵雨",
    "Lightning": "闪电",
    "Light rain with thunderstorm": "雷阵雨",
    "Patchy light snow": "局部小雪",
    "Blowing snow": "吹雪",
    "Blizzard": "暴风雪",
    "Patchy freezing drizzle nearby": "局部冻毛毛雨",
    "Light freezing rain": "小冻雨",
    "Moderate or heavy freezing rain": "中到大冻雨",
    "Light sleet": "小冰雹",
    "Moderate or heavy sleet": "中到大冰雹",
    "Patchy sleet nearby": "局部冰雹",
    "Light shower snow": "小阵雪",
    "Patchy moderate snow": "局部中雪",
    "Patchy heavy snow": "局部大雪",
    "Windy": "大风",
    "Calm": "平静",
    "Heavy freezing drizzle": "大冻毛毛雨",
}

# ─── AQI 等级定义 ─────────────────────────────────────────
AQI_LEVELS = {
    (0, 50): {
        "level": "优",
        "color": "绿色",
        "emoji": "😊",
        "desc": "空气清新，非常适合户外活动",
    },
    (51, 100): {
        "level": "良",
        "color": "黄色",
        "emoji": "🙂",
        "desc": "空气质量可以接受，敏感人群应减少长时间户外高强度运动",
    },
    (101, 150): {
        "level": "轻度污染",
        "color": "橙色",
        "emoji": "😐",
        "desc": "敏感人群应减少户外活动，一般人群适量减少户外运动",
    },
    (151, 200): {
        "level": "中度污染",
        "color": "红色",
        "emoji": "😷",
        "desc": "建议佩戴口罩，减少户外活动，特别是老人和儿童",
    },
    (201, 300): {
        "level": "重度污染",
        "color": "紫色",
        "emoji": "😨",
        "desc": "请避免户外活动，关闭门窗，使用空气净化器",
    },
    (301, 500): {
        "level": "严重污染",
        "color": "褐红色",
        "emoji": "😱",
        "desc": "健康警报！请留在室内，必须外出时请佩戴防护口罩",
    },
}

# ─── 站点列表（用于数据导入和训练） ──────────────────────
TARGET_SITES = ["2610A", "2611A", "2596A", "1916A", "1917A", "3376A", "3377A"]
