"""
天气预报服务 — 通过 wttr.in 免费 API 获取实时天气信息
"""
import requests
from config import LOCATION_CITY_MAP, WEATHER_ZH_MAP


def get_weather_info(location: str) -> dict | None:
    """通过 wttr.in 获取指定地区的实时天气信息（免费，无需 API Key）

    Args:
        location: 地区名称（如"曲靖"、"昭通"、"文山州"）

    Returns:
        dict: 天气信息字典，包含 temp, feels_like, humidity, weather,
              wind_speed, wind_dir, max_temp, min_temp；失败返回 None
    """
    city = LOCATION_CITY_MAP.get(location, location)
    try:
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data["current_condition"][0]
        today = data["weather"][0]

        temp = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]

        weather_en = current["weatherDesc"][0]["value"]
        weather_zh = current.get("lang_zh", [{}])[0].get("value", "")
        if not weather_zh or weather_zh == weather_en:
            weather_zh = WEATHER_ZH_MAP.get(weather_en, weather_en)

        wind_speed = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        max_temp = today["maxtempC"]
        min_temp = today["mintempC"]

        return {
            "temp": temp,
            "feels_like": feels_like,
            "humidity": humidity,
            "weather": weather_zh,
            "wind_speed": wind_speed,
            "wind_dir": wind_dir,
            "max_temp": max_temp,
            "min_temp": min_temp,
        }
    except Exception as e:
        print(f"⚠️ 获取 {city} 天气失败: {e}")
        return None
