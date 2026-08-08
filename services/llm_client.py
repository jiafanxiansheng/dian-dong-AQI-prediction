"""
DeepSeek LLM 对话客户端 — 根据 AQI 预测数据生成自然语言回复
"""
import requests
from datetime import datetime
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_URL,
    PRIMARY_FORECAST_HOURS,
    SITE_DETAILS,
)
from services.predictor import get_aqi_info


def call_deepseek_api(
    user_query: str,
    aqi_info: dict,
    location: str,
    predicted_aqi: float,
    site_name: str | None = None,
    weather_info: dict | None = None,
    multi_preds: dict | None = None,
    is_unsupported: bool = False,
    is_unknown_location: bool = False,
) -> str:
    """调用 DeepSeek API 生成有情感的空气质量助手回复

    Args:
        user_query: 用户原始提问
        aqi_info: AQI 等级信息字典（含 level, emoji, desc, time）
        location: 地区名称
        predicted_aqi: 预测的 AQI 数值
        site_name: 站点名称（可选）
        weather_info: 当前天气信息（可选）
        multi_preds: 分步预测数据字典（可选）
        is_unsupported: 是否为不支持的地区
        is_unknown_location: 是否未识别出地点

    Returns:
        str: AI 生成的回复文本
    """
    aqi_level = aqi_info["level"]
    emoji = aqi_info["emoji"]
    desc = aqi_info["desc"]

    try:
        time_str = aqi_info["time"].strftime("%m月%d日 %H:%M")
    except Exception:
        time_str = f"未来{PRIMARY_FORECAST_HOURS}小时"

    # ─── 构造系统提示词 ───────────────────────────────
    if is_unknown_location:
        system_prompt = _build_unknown_location_prompt(user_query)
    elif is_unsupported:
        system_prompt = _build_unsupported_prompt(
            user_query, location, site_name, weather_info
        )
    else:
        system_prompt = _build_normal_prompt(
            user_query, location, site_name, predicted_aqi, aqi_level,
            emoji, desc, time_str, weather_info, multi_preds
        )

    # ─── 调用 API ─────────────────────────────────────
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            "temperature": 0.9,
            "max_tokens": 300,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3,
        }

        response = requests.post(
            DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"⚠️ API 调用失败: {e}")
        # 降级：基于 AQI 数据直接生成回复，不依赖 LLM
        return _build_fallback_response(
            user_query, location, site_name, predicted_aqi,
            aqi_level, emoji, desc, time_str, weather_info,
            multi_preds, is_unsupported, is_unknown_location,
        )


# ═══════════════════════════════════════════════════════════
#  提示词构造器（内部函数）
# ═══════════════════════════════════════════════════════════

def _build_unknown_location_prompt(user_query: str) -> str:
    sites_desc = "\n".join(
        f"  * {loc}：{', '.join(info['name'] for code, info in SITE_DETAILS.items() if info['location'] == loc)}"
        for loc in ["曲靖", "昭通", "文山州"]
    )
    return f"""你是一个贴心的空气质量助手，名叫"清新小助手"。

当前情况：
- 用户的提问中没有明确的地点信息
- 我们目前仅支持滇东地区：曲靖、昭通、文山州
- 系统覆盖的具体站点包括：
{sites_desc}

回复要求：
- 友好地引导用户明确询问具体的地点或站点
- 可以介绍系统支持的地区和站点列表
- 如果用户问的是其他问题（如问候、感谢等），自然回应
- 语气温暖友好，像朋友聊天一样
- 可以适当加入一些emoji增加亲和力
- 回复要简洁自然，不要过于机械

用户问题：{user_query}

请用温暖、自然的语言回答："""


def _build_unsupported_prompt(
    user_query: str,
    location: str,
    site_name: str | None,
    weather_info: dict | None,
) -> str:
    site_line = f"- 监测站点：{site_name}\n" if site_name else ""
    weather_text = _format_weather_text(weather_info)
    return f"""你是一个贴心的空气质量助手，名叫"清新小助手"。

当前情况：
- 用户询问了关于{location}的空气质量
- 但{location}不在我们系统的预测覆盖范围内
- 我们目前仅支持滇东地区：曲靖、昭通、文山州
{weather_text}
回复要求：
- 友好地告知用户该地区暂不在预测覆盖范围内
- 说明系统目前支持滇东地区（曲靖、昭通、文山州）的空气质量预测
- 如果有天气数据，可以分享当前天气信息作为参考
- 给出一些通用的空气质量关注建议
- 引导用户尝试查询支持的地区
- 语气温暖友好，不要让用户感到失望

用户问题：{user_query}

请用温暖、自然的语言回答："""


def _build_normal_prompt(
    user_query: str,
    location: str,
    site_name: str | None,
    predicted_aqi: float,
    aqi_level: str,
    emoji: str,
    desc: str,
    time_str: str,
    weather_info: dict | None,
    multi_preds: dict | None,
) -> str:
    # 风格指引
    if predicted_aqi <= 50:
        style_guide = "\n回复风格建议：\n- 可以活泼欢快一些，表达对好天气的喜悦\n- 鼓励用户多出门活动，享受美好时光\n- 可以加入一些生活化的场景描述（如散步、跑步、郊游等）\n- 语气轻松愉快，像好朋友分享好消息\n"
    elif predicted_aqi <= 100:
        style_guide = "\n回复风格建议：\n- 语气温和友好，带点轻松的提醒\n- 可以提到适合的活动，但也给出小贴士\n- 像朋友之间的日常聊天，自然随意\n- 可以适当加入一些关怀的语气\n"
    elif predicted_aqi <= 150:
        style_guide = "\n回复风格建议：\n- 语气带点关切，但不要过于紧张\n- 给出实用的建议，但不要太严肃\n- 可以提到一些室内活动的建议\n- 像朋友之间的善意提醒\n"
    elif predicted_aqi <= 200:
        style_guide = "\n回复风格建议：\n- 语气认真但不恐慌，表达真诚的关心\n- 给出具体的防护建议\n- 可以提到一些室内活动或替代方案\n- 像家人一样的叮嘱和关怀\n"
    else:
        style_guide = "\n回复风格建议：\n- 语气紧急但不制造恐慌，表达深切的关心\n- 强调防护措施的重要性\n- 给出明确的行动建议\n- 像好朋友在关键时刻的提醒\n"

    site_line = f"- 监测站点：{site_name}\n" if site_name else ""
    weather_text = _format_weather_text(weather_info)

    # 多步预测详情
    multi_step_text = ""
    if multi_preds:
        steps = []
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_aqi_info = get_aqi_info(pred["aqi"])
            try:
                pred_time = pred["time"]
                step_time_str = pred_time.strftime("%m月%d日 %H:%M")
            except Exception:
                step_time_str = "时间未知"
            steps.append(
                f"  - 未来{fh}小时（{step_time_str}）：AQI {pred['aqi']:.0f}（{step_aqi_info['level']}）"
            )
        multi_step_text = "- 分步预测详情：\n" + "\n".join(steps) + "\n"

    effective_fh = max(multi_preds.keys()) if multi_preds else PRIMARY_FORECAST_HOURS
    forecast_desc = f"从当前时间起，预测未来{effective_fh}小时后的空气质量（目标时间：{time_str}）"

    return f"""你是一个贴心的空气质量助手，名叫"清新小助手"。你的任务是根据AQI预测数据和天气信息，用温暖、有感情的语言回答用户的问题。

当前信息：
- 地点：{location}
{site_line}- 预测说明：{forecast_desc}
- 预测AQI：{predicted_aqi:.0f}
- 空气质量等级：{aqi_level} {emoji}
- 健康建议：{desc}
{weather_text}{multi_step_text}
{style_guide}

重要要求：
- 回复中必须明确提到预测的目标时间（{time_str}），让用户知道这是对未来哪个时间点的预测
- 必须提到这是未来{effective_fh}小时的预测结果
- 如果有分步预测数据，请简要提及空气质量的变化趋势（如逐渐好转或恶化）
- 请结合当前天气和气温情况给出综合建议（如穿衣、出行等）

你可以自由发挥，不一定要严格按照模板。可以：
- 使用不同的表达方式和句式
- 适当变化emoji的使用
- 根据用户的提问方式调整回复风格
- 如果用户问得比较随意，可以更轻松地回应
- 如果用户问得很正式，可以稍微正式一些
- 可以加入一些生活化的比喻或场景
- 不必每次都提到所有信息，重点突出最关键的内容

用户问题：{user_query}

请用温暖、自然的语言回答："""


def _format_weather_text(weather_info: dict | None) -> str:
    """格式化天气信息为提示词文本"""
    if not weather_info:
        return "- 当前天气：暂无天气数据\n"
    return (
        f"- 当前天气：{weather_info['weather']}，气温{weather_info['temp']}°C"
        f"（体感{weather_info['feels_like']}°C）\n"
        f"- 今日温度范围：{weather_info['min_temp']}°C ~ {weather_info['max_temp']}°C\n"
        f"- 湿度：{weather_info['humidity']}% | "
        f"风速：{weather_info['wind_speed']}km/h {weather_info['wind_dir']}\n"
    )


# ═══════════════════════════════════════════════════════════
#  降级回复：LLM 不可用时，基于 AQI 数据直接生成回复
# ═══════════════════════════════════════════════════════════

def _build_fallback_response(
    user_query: str,
    location: str,
    site_name: str | None,
    predicted_aqi: float,
    aqi_level: str,
    emoji: str,
    desc: str,
    time_str: str,
    weather_info: dict | None,
    multi_preds: dict | None,
    is_unsupported: bool = False,
    is_unknown_location: bool = False,
) -> str:
    """在 LLM API 不可用时，用模板生成自然语言回复"""

    # ── 未知地点 ──
    if is_unknown_location:
        sites_list = "、".join(
            f"{loc}（{'/'.join(info['name'] for code, info in __import__('config').SITE_DETAILS.items() if info['location'] == loc)}）"
            for loc in ["曲靖", "昭通", "文山州"]
        )
        return (
            f"你好！😊 我没有从你的问题中识别到具体地点。\n\n"
            f"我目前支持查询滇东以下地区的空气质量：\n{sites_list}\n\n"
            f"你可以试试问我：\n"
            f'• "曲靖今天空气如何？"\n'
            f'• "曲靖师范学院空气质量怎么样？"\n'
            f'• "文山的空气质量怎么样？"'
        )

    # ── 不支持地区 ──
    if is_unsupported:
        weather_line = ""
        if weather_info:
            weather_line = (
                f"\n\n不过我可以告诉你{location}当前的天气：\n"
                f"🌤️ {weather_info['weather']}，气温{weather_info['temp']}°C"
                f"（体感{weather_info['feels_like']}°C），"
                f"湿度{weather_info['humidity']}%。"
            )
        return (
            f"抱歉，{location}暂时不在我的AQI预测覆盖范围内。😅\n\n"
            f"我目前只支持滇东地区：曲靖、昭通、文山州。{weather_line}\n\n"
            f"你可以试试查询曲靖、昭通或文山的空气质量～"
        )

    # ── 正常预测回复 ──
    name_line = f"「{site_name}」" if site_name else location
    lines = [f"{emoji} {name_line}的空气质量预报来啦！\n"]

    # 主预报
    lines.append(f"📊 预测目标时间：{time_str}")
    lines.append(f"🎯 AQI 预测值：{predicted_aqi:.0f} — {aqi_level}")
    lines.append(f"💡 {desc}")

    # 分步预测
    if multi_preds and len(multi_preds) > 1:
        lines.append(f"\n📈 分时预报：")
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            try:
                t = pred["time"].strftime("%m月%d日 %H:%M")
            except Exception:
                t = f"未来{fh}h"
            from services.predictor import get_aqi_info
            si = get_aqi_info(pred["aqi"])
            trend = ""
            lines.append(f"  · 未来{fh}h（{t}）：AQI {pred['aqi']:.0f} {si['emoji']} {si['level']}")

    # 天气
    if weather_info:
        lines.append(
            f"\n🌤️ 当前天气：{weather_info['weather']}，气温{weather_info['temp']}°C"
            f"（体感{weather_info['feels_like']}°C）"
        )
        lines.append(
            f"🌡️ 今日温度：{weather_info['min_temp']}°C ~ {weather_info['max_temp']}°C"
            f"  |  💧 湿度：{weather_info['humidity']}%"
        )

    # 生活建议
    if predicted_aqi <= 50:
        lines.append(f"\n☀️ 空气质量很好，适合出门活动，享受新鲜空气吧！")
    elif predicted_aqi <= 100:
        lines.append(f"\n🌤️ 空气质量还行，正常户外活动没问题～")
    elif predicted_aqi <= 150:
        lines.append(f"\n⚠️ 敏感人群建议减少户外活动，出门可戴口罩。")
    else:
        lines.append(f"\n🚨 空气不太好，建议减少外出，关好门窗，出门记得戴好口罩！")

    return "\n".join(lines)
