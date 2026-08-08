"""
清新小助手 — 滇东AQI智能预测系统 Web 服务入口
启动: python app.py
访问: http://localhost:5000
"""
import os
import sys
import time
import traceback
import logging
from datetime import datetime

from flask import Flask, render_template, request, jsonify

# 确保项目根在 sys.path 中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import (
    FORECAST_HOURS_LIST,
    PRIMARY_FORECAST_HOURS,
    SITE_DETAILS,
    AQI_LEVELS,
    SUPPORTED_LOCATIONS,
)
from services.predictor import (
    load_models,
    predict_aqi,
    predict_location_aqi,
    get_aqi_info,
    ProphetModelWrapper,
    CatBoostModelWrapper,
)
from services.weather import get_weather_info
from services.intent_parser import parse_user_intent
from services.llm_client import call_deepseek_api

import numpy as np

# ─── 日志配置 ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qingxin")

# ─── Flask 初始化 ────────────────────────────────────────
app = Flask(__name__)

# ─── 全局模型存储 ────────────────────────────────────────
models = {}


def init_models():
    """启动时加载所有预测模型"""
    global models
    logger.info("正在加载空气质量预测模型...")
    models = load_models()

    # 统计各类型模型数量
    prophet_count = 0
    catboost_count = 0
    for site_models in models.values():
        for m in site_models.values():
            if isinstance(m, CatBoostModelWrapper):
                catboost_count += 1
            elif isinstance(m, ProphetModelWrapper):
                prophet_count += 1

    total_sites = sum(1 for m in models.values() if m)
    logger.info(
        f"[OK] 成功加载 {total_sites} 个站点模型 "
        f"(Prophet: {prophet_count}, CatBoost: {catboost_count})"
    )
    return models


# ═══════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Web 聊天界面"""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """健康检查 / 模型状态"""
    loaded_sites = []
    model_summary = {}
    for site, fh_models in models.items():
        if fh_models:
            site_info = {
                "code": site,
                "name": SITE_DETAILS[site]["name"],
                "location": SITE_DETAILS[site]["location"],
                "models": {},
            }
            for fh, m in fh_models.items():
                model_type = (
                    "catboost" if isinstance(m, CatBoostModelWrapper) else "prophet"
                )
                site_info["models"][str(fh)] = model_type
                model_summary[model_type] = model_summary.get(model_type, 0) + 1
            loaded_sites.append(site_info)

    return jsonify({
        "success": True,
        "models_loaded": len(loaded_sites),
        "forecast_hours": FORECAST_HOURS_LIST,
        "primary_forecast_hours": PRIMARY_FORECAST_HOURS,
        "supported_locations": sorted(SUPPORTED_LOCATIONS),
        "sites": loaded_sites,
        "model_summary": model_summary,
        "status": "running",
        "uptime": time.time() - app.config.get("START_TIME", time.time()),
    })


@app.route("/api/sites", methods=["GET"])
def api_sites():
    """获取所有站点信息"""
    sites_list = []
    for code, info in SITE_DETAILS.items():
        has_models = code in models and bool(models[code])
        model_types = {}
        if has_models:
            for fh, m in models[code].items():
                mt = "catboost" if isinstance(m, CatBoostModelWrapper) else "prophet"
                model_types[str(fh)] = mt
        sites_list.append({
            "code": code,
            "name": info["name"],
            "location": info["location"],
            "has_models": has_models,
            "model_types": model_types,
        })

    return jsonify({
        "success": True,
        "sites": sites_list,
        "locations": sorted(SUPPORTED_LOCATIONS),
    })


@app.route("/api/aqi-levels", methods=["GET"])
def api_aqi_levels():
    """获取 AQI 等级定义"""
    levels = []
    for (min_val, max_val), info in AQI_LEVELS.items():
        levels.append({
            "min": min_val,
            "max": max_val,
            "level": info["level"],
            "emoji": info["emoji"],
            "desc": info["desc"],
            "color": info["color"],
        })
    return jsonify({"success": True, "levels": levels})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """自由对话接口 — 用户输入自然语言，返回 AI 回复 + 结构化数据"""
    t0 = time.time()
    try:
        data = request.get_json(force=True) if request.data else {}
        user_input = (data.get("message", "") or "").strip()

        if not user_input:
            return jsonify({"success": False, "error": "请输入您的问题"}), 400

        if len(user_input) > 500:
            return jsonify({"success": False, "error": "输入内容过长，请控制在500字以内"}), 400

        result = chat_with_assistant(user_input)

        elapsed = time.time() - t0
        logger.info(f"请求完成 [{elapsed:.2f}s]: {user_input[:50]}...")

        return jsonify({
            "success": True,
            "response": result["response"],
            "data": result.get("data"),
            "elapsed_ms": round(elapsed * 1000),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception:
        traceback.print_exc()
        logger.error(f"请求失败: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "服务暂时不可用，请稍后重试",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }), 500


@app.route("/api/quick-query", methods=["POST"])
def api_quick_query():
    """快捷查询接口 — 按地区/站点名快速查询"""
    t0 = time.time()
    try:
        data = request.get_json(force=True) if request.data else {}
        location = (data.get("location", "") or "").strip()
        site_name = (data.get("site_name", "") or "").strip()

        if site_name:
            query = f"{site_name}的空气质量如何？"
        elif location:
            query = f"{location}的空气质量怎么样？"
        else:
            query = "曲靖今天空气如何？"

        result = chat_with_assistant(query)

        elapsed = time.time() - t0
        return jsonify({
            "success": True,
            "response": result["response"],
            "data": result.get("data"),
            "query": query,
            "elapsed_ms": round(elapsed * 1000),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "服务暂时不可用，请稍后重试",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }), 500


# ═══════════════════════════════════════════════════════════
#  核心对话逻辑
# ═══════════════════════════════════════════════════════════

def chat_with_assistant(user_input: str) -> dict:
    """与空气质量助手对话，返回回复和数据

    支持三种模式：
    - site:    查询具体站点
    - location: 查询整个地区（所有站点综合）
    - unsupported: 不支持的地区
    """
    predict_type, target_code, target_name = parse_user_intent(user_input)

    # ── 不支持的地区 ──────────────────────────────────
    if predict_type == "unsupported":
        logger.info(f"查询不支持地区: {target_name}")
        weather_info = get_weather_info(target_name)
        if weather_info:
            logger.info(f"  {target_name}天气: {weather_info['weather']} {weather_info['temp']}°C")

        aqi_info = {
            "level": "未知", "emoji": "🌍",
            "desc": "暂无该地区空气质量预测数据",
            "time": datetime.now(),
        }
        response = call_deepseek_api(
            user_input, aqi_info, target_name, 0,
            weather_info=weather_info, is_unsupported=True,
        )
        return {
            "response": response,
            "data": {
                "aqi": None, "level": "未知", "emoji": "🌍",
                "location": target_name, "site_name": None,
                "steps": [], "weather": weather_info,
            },
        }

    # ── 具体站点 ──────────────────────────────────────
    if predict_type == "site":
        logger.info(f"查询站点: {target_name} ({target_code})")
        prediction = predict_aqi(target_code, models)

        if prediction is None:
            return {
                "response": f"抱歉，暂时无法获取{target_name}站点的空气质量数据，请稍后再试～",
                "data": None,
            }

        multi_preds = prediction["multi_preds"]

        if PRIMARY_FORECAST_HOURS not in multi_preds:
            return {
                "response": f"抱歉，{target_name}站点的{PRIMARY_FORECAST_HOURS}h预测模型未加载，请稍后再试～",
                "data": None,
            }

        primary_pred = multi_preds[PRIMARY_FORECAST_HOURS]
        predicted_aqi = primary_pred["aqi"]
        aqi_info = get_aqi_info(predicted_aqi)
        aqi_info["time"] = primary_pred["time"]

        location = SITE_DETAILS[target_code]["location"]
        weather_info = get_weather_info(location)

        # 日志输出
        logger.info("分步预测结果:")
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_info = get_aqi_info(pred["aqi"])
            logger.info(
                f"  +{fh}h ({pred['time'].strftime('%m/%d %H:%M')}): "
                f"AQI={pred['aqi']:.0f} ({step_info['level']})"
            )
        if weather_info:
            logger.info(f"  天气: {weather_info['weather']} {weather_info['temp']}°C")

        steps = []
        for fh in sorted(multi_preds.keys()):
            pred = multi_preds[fh]
            step_info = get_aqi_info(pred["aqi"])
            steps.append({
                "hours": fh,
                "time": pred["time"].strftime("%m月%d日 %H:%M"),
                "aqi": pred["aqi"],
                "level": step_info["level"],
                "emoji": step_info["emoji"],
            })

        response = call_deepseek_api(
            user_input, aqi_info, location, predicted_aqi,
            site_name=target_name, weather_info=weather_info,
            multi_preds=multi_preds,
        )

        return {
            "response": response,
            "data": {
                "aqi": predicted_aqi,
                "level": aqi_info["level"],
                "emoji": aqi_info["emoji"],
                "location": location,
                "site_name": target_name,
                "steps": steps,
                "weather": weather_info,
            },
        }

    # ── 地区综合 ──────────────────────────────────────
    else:
        logger.info(f"查询地区: {target_name}")
        prediction = predict_location_aqi(target_name, models)

        if prediction is None:
            return {
                "response": f"抱歉，暂时无法获取{target_name}地区的空气质量数据，请稍后再试～",
                "data": None,
            }

        weather_info = get_weather_info(target_name)

        logger.info(f"综合预测结果（{prediction['site_count']}个站点）:")
        for site_pred in prediction["sites"]:
            logger.info(f"  📍 {site_pred['site_name']}:")
            for fh in sorted(site_pred["multi_preds"].keys()):
                pred = site_pred["multi_preds"][fh]
                step_info = get_aqi_info(pred["aqi"])
                logger.info(
                    f"    +{fh}h ({pred['time'].strftime('%m/%d %H:%M')}): "
                    f"AQI={pred['aqi']:.0f} ({step_info['level']})"
                )
        if weather_info:
            logger.info(
                f"  天气: {weather_info['weather']} {weather_info['temp']}°C "
                f"({weather_info['min_temp']}~{weather_info['max_temp']}°C)"
            )

        # 计算地区平均 AQI
        has_primary = any(
            PRIMARY_FORECAST_HOURS in s["multi_preds"] for s in prediction["sites"]
        )
        if has_primary:
            avg_aqi = np.mean([
                s["multi_preds"][PRIMARY_FORECAST_HOURS]["aqi"]
                for s in prediction["sites"]
                if PRIMARY_FORECAST_HOURS in s["multi_preds"]
            ])
            ref_site = next(
                s for s in prediction["sites"]
                if PRIMARY_FORECAST_HOURS in s["multi_preds"]
            )
            aqi_info = get_aqi_info(avg_aqi)
            aqi_info["time"] = ref_site["multi_preds"][PRIMARY_FORECAST_HOURS]["time"]
        else:
            all_fh = set()
            for s in prediction["sites"]:
                all_fh.update(s["multi_preds"].keys())
            actual_fh = sorted(all_fh)[0] if all_fh else PRIMARY_FORECAST_HOURS
            avg_aqi = np.mean([
                s["multi_preds"][actual_fh]["aqi"]
                for s in prediction["sites"]
                if actual_fh in s["multi_preds"]
            ])
            ref_site = next(
                (s for s in prediction["sites"] if actual_fh in s["multi_preds"]), None
            )
            aqi_info = get_aqi_info(avg_aqi)
            aqi_info["time"] = (
                ref_site["multi_preds"][actual_fh]["time"]
                if ref_site else datetime.now()
            )

        # 构建分步汇总
        avg_steps = []
        for fh in FORECAST_HOURS_LIST:
            if any(fh in s["multi_preds"] for s in prediction["sites"]):
                avg_step_aqi = round(np.mean([
                    s["multi_preds"][fh]["aqi"]
                    for s in prediction["sites"]
                    if fh in s["multi_preds"]
                ]), 1)
                step_info = get_aqi_info(avg_step_aqi)
                ref_s = next(
                    (s for s in prediction["sites"] if fh in s["multi_preds"]), None
                )
                ref_time_str = (
                    ref_s["multi_preds"][fh]["time"].strftime("%m月%d日 %H:%M")
                    if ref_s else ""
                )
                avg_steps.append({
                    "hours": fh,
                    "time": ref_time_str,
                    "aqi": avg_step_aqi,
                    "level": step_info["level"],
                    "emoji": step_info["emoji"],
                })

        # 各站点详情
        site_details = []
        for site_pred in prediction["sites"]:
            site_steps = []
            for fh in sorted(site_pred["multi_preds"].keys()):
                pred = site_pred["multi_preds"][fh]
                si = get_aqi_info(pred["aqi"])
                site_steps.append({"hours": fh, "aqi": pred["aqi"], "level": si["level"]})
            site_details.append({"name": site_pred["site_name"], "steps": site_steps})

        # 构建 API 用 multi_preds
        _multi_preds_for_api = {}
        for fh in FORECAST_HOURS_LIST:
            sites_with_fh = [s for s in prediction["sites"] if fh in s["multi_preds"]]
            if sites_with_fh:
                avg_aqi_fh = round(np.mean([
                    s["multi_preds"][fh]["aqi"] for s in sites_with_fh
                ]), 1)
                ref_time = sites_with_fh[0]["multi_preds"][fh]["time"]
                _multi_preds_for_api[fh] = {"aqi": avg_aqi_fh, "time": ref_time}

        response = call_deepseek_api(
            user_input, aqi_info, target_name, avg_aqi,
            site_name=f"综合{prediction['site_count']}个站点",
            weather_info=weather_info,
            multi_preds=_multi_preds_for_api,
        )

        return {
            "response": response,
            "data": {
                "aqi": round(float(avg_aqi), 1),
                "level": aqi_info["level"],
                "emoji": aqi_info["emoji"],
                "location": target_name,
                "site_name": f"综合{prediction['site_count']}个站点",
                "steps": avg_steps,
                "sites": site_details,
                "weather": weather_info,
            },
        }


# ═══════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  清新小助手 - 滇东空气质量智能预测系统 v2.0")
    print("=" * 60)

    app.config["START_TIME"] = time.time()
    init_models()

    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("FLASK_PORT", "5000"))

    print(f"\n>> 服务启动中...")
    print(f"   访问地址: http://localhost:{port}")
    print(f"   API 状态: http://localhost:{port}/api/status")
    print(f"   Debug 模式: {'开启' if debug_mode else '关闭'}")
    print("=" * 60)

    app.run(debug=debug_mode, host="0.0.0.0", port=port)
