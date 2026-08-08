"""
实时空气质量数据获取服务 — 基于 air.cnemc.cn 官方 API + aqicn.org 降级

数据源：
  1. air.cnemc.cn API — /HourChangesPublish/GetAqiHistoryByCondition（主源）
  2. aqicn.org 公开 API — demo token（备用源）
"""
import time
import re
import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

from config import DB_CONFIG, SITE_DETAILS

# ─── 站点编码映射（项目编码 → air.cnemc.cn 站点编码） ──────
# 部分站点编码可能不一致，需做映射
SITE_CODE_MAP = {
    # 曲靖 (4个站点)
    "1916A": "1916A",   # 环境监测站
    "1917A": "1917A",   # 烟厂办公区
    "3376A": "3376A",   # 南苑二区
    "3377A": "3377A",   # 曲靖师范学院
    # 昭通 (1个站点)
    "2596A": "2596A",   # 监测站
    # 文山 (2个站点)
    "2610A": "2610A",   # 州水务局
    "2611A": "2611A",   # 市便民服务中心
}

# air.cnemc.cn API 基础地址
AIRCNEMC_BASE = "https://air.cnemc.cn:18007"
AIRCNEMC_API = f"{AIRCNEMC_BASE}/HourChangesPublish/GetAqiHistoryByCondition"

# HTTP 请求头（模拟正常浏览器）
API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36 Edg/149.0.0.0"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": AIRCNEMC_BASE,
    "Referer": f"{AIRCNEMC_BASE}/",
}

# 站点到城市的映射（用于 aqicn.org 降级）
LOCATION_CITY_MAP = {
    "曲靖": "qujing",
    "昭通": "zhaotong",
    "文山州": "wenshan",
    "文山": "wenshan",
}

# 创建持久 Session（复用 TCP 连接 + Cookie）
_session = None


def _get_session() -> requests.Session:
    """获取或创建 HTTP Session（懒加载，模块级复用）"""
    global _session
    if _session is None:
        _session = requests.Session()
        # 预热：访问首页，获取必要的 Cookie
        try:
            _session.get(
                AIRCNEMC_BASE + "/",
                headers={
                    "User-Agent": API_HEADERS["User-Agent"],
                    "Accept": "text/html,application/xhtml+xml,*/*",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                },
                timeout=15,
                verify=False,
            )
        except Exception:
            pass
    return _session


# ═══════════════════════════════════════════════════════════
#  时间戳解析
# ═══════════════════════════════════════════════════════════

def _parse_timestamp(time_str: str) -> datetime | None:
    """解析 API 返回的 ASP.NET 时间戳格式 /Date(毫秒数)/"""
    if not time_str:
        return None
    match = re.search(r"/Date\((\d+)\)/", str(time_str))
    if match:
        timestamp_ms = int(match.group(1))
        return datetime.fromtimestamp(timestamp_ms / 1000)
    # 尝试直接解析 ISO 格式
    try:
        return pd.to_datetime(time_str).to_pydatetime()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
#  方法1: air.cnemc.cn 官方 API（主要数据源）
# ═══════════════════════════════════════════════════════════

def fetch_via_aircnemc(site_code: str) -> pd.DataFrame | None:
    """通过 air.cnemc.cn 官方 API 获取站点历史 AQI 数据

    API: POST /HourChangesPublish/GetAqiHistoryByCondition
    参数: stationCode={编码}
    返回: JSON 数组，每项含 TimePoint, AQI, PM2_5, PM10, SO2, NO2, CO, O3 等
    """
    t0 = time.time()

    try:
        code = SITE_CODE_MAP.get(site_code, site_code)
        session = _get_session()

        print(f"  [FETCH] air.cnemc.cn API 获取站点 {site_code} 数据...")

        resp = session.post(
            AIRCNEMC_API,
            params={"stationCode": code},
            headers=API_HEADERS,
            timeout=15,
            verify=False,
        )

        if resp.status_code != 200:
            print(f"  [WARN] API 返回 HTTP {resp.status_code}")
            return None

        # 尝试解析 JSON
        try:
            data = resp.json()
        except Exception:
            print(f"  [WARN] API 响应非 JSON 格式")
            return None

        if not isinstance(data, list) or len(data) == 0:
            print(f"  [WARN] API 返回空数据（站点 {site_code} 可能已变更编码）")
            return None

        # 解析每条记录
        records = []
        for item in data:
            ts = _parse_timestamp(item.get("TimePoint", ""))
            if ts is None:
                continue

            aqi_val = item.get("AQI")
            if aqi_val is None or aqi_val == "" or aqi_val == "NA":
                continue

            record = {
                "datetime": ts,
                "AQI": float(aqi_val),
            }

            # 提取各污染物
            pollutant_map = {
                "PM2_5": "PM2.5",
                "PM10": "PM10",
                "SO2": "SO2",
                "NO2": "NO2",
                "CO": "CO",
                "O3": "O3",
            }
            for api_key, db_key in pollutant_map.items():
                val = item.get(api_key)
                if val is not None and val != "" and val != "NA":
                    try:
                        record[db_key] = float(val)
                    except (ValueError, TypeError):
                        pass

            records.append(record)

        if not records:
            print("  [WARN] 解析后无有效数据")
            return None

        df = pd.DataFrame(records).drop_duplicates(subset=["datetime"])
        df = df.sort_values("datetime")

        elapsed = time.time() - t0
        print(f"  [OK] air.cnemc.cn 获取 {len(df)} 条数据（{elapsed:.1f}s）")
        return df

    except requests.exceptions.Timeout:
        print(f"  [WARN] API 请求超时（{time.time() - t0:.1f}s）")
        return None
    except Exception as e:
        print(f"  [WARN] air.cnemc.cn 请求失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  方法2: aqicn.org API（备用数据源）
# ═══════════════════════════════════════════════════════════

def fetch_via_aqicn(location: str) -> pd.DataFrame | None:
    """通过 aqicn.org 公开 API 获取城市 AQI（备用）"""
    import requests as req

    try:
        city = LOCATION_CITY_MAP.get(location, location.lower())
        url = f"https://api.waqi.info/feed/{city}/?token=demo"

        print(f"  [FETCH] aqicn.org 备用源获取 {location} 数据...")
        resp = req.get(url, timeout=10)

        if resp.status_code != 200:
            print(f"  [WARN] aqicn.org HTTP {resp.status_code}")
            return None

        content_type = resp.headers.get("content-type", "")
        if "json" not in content_type:
            print(f"  [WARN] aqicn.org 非 JSON 响应")
            return None

        data = resp.json()
        if not isinstance(data, dict) or data.get("status") != "ok":
            print(f"  [WARN] aqicn.org 返回错误: {data.get('status', 'unknown')}")
            return None

        aqi_data = data.get("data")
        if not aqi_data or not aqi_data.get("aqi"):
            print("  [WARN] aqicn.org 无 AQI 数据")
            return None

        aqi_val = float(aqi_data["aqi"])
        iaqi = aqi_data.get("iaqi", {})

        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        record = {"datetime": now, "AQI": aqi_val}

        for key in ["pm25", "pm10", "o3", "no2", "so2", "co"]:
            if key in iaqi:
                record[key.upper().replace("25", "2.5")] = float(iaqi[key]["v"])

        df = pd.DataFrame([record])
        print(f"  [OK] aqicn.org 获取到 {location} AQI={aqi_val}")
        return df

    except req.exceptions.Timeout:
        print("  [WARN] aqicn.org 请求超时")
        return None
    except Exception as e:
        print(f"  [WARN] aqicn.org 请求失败: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  数据获取入口（多源降级）
# ═══════════════════════════════════════════════════════════

def fetch_realtime_data(
    site_code: str, location: str, headless: bool = True
) -> pd.DataFrame | None:
    """获取实时空气质量数据（多源降级策略）

    策略：
      1. air.cnemc.cn 官方 API — 返回最近24小时数据
      2. aqicn.org 公开 API — 返回当前单点数据
    """
    # 策略1: air.cnemc.cn 官方 API
    df = fetch_via_aircnemc(site_code)
    if df is not None and len(df) > 0:
        return df
    print("  [WARN] air.cnemc.cn API 失败，尝试备用数据源...")

    # 策略2: aqicn.org 备用
    df = fetch_via_aqicn(location)
    if df is not None and len(df) > 0:
        return df

    print("[ERROR] 所有数据源均失败，无法获取实时数据")
    return None


# ═══════════════════════════════════════════════════════════
#  兼容旧接口
# ═══════════════════════════════════════════════════════════

def check_webdriver_available() -> bool:
    """WebDriver 检查（已不依赖 Selenium，始终返回 True）"""
    return True


def parse_chinese_datetime(time_str: str) -> datetime | None:
    """解析中文时间格式（兼容旧接口）"""
    try:
        match = re.search(r"(\d+)时", str(time_str))
        if match:
            hour = int(match.group(1))
            now = datetime.now()
            return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        return None
    except Exception:
        return None


def integrate_data(all_data: dict, location: str) -> pd.DataFrame:
    """整合各污染物数据为统一的 DataFrame（兼容旧接口）"""
    from services.data_fetcher import _parse_timestamp as _pt

    data_dict = {"datetime": [], "AQI": []}
    pollutant_names = ["AQI", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    if "AQI" in all_data:
        for t, v in zip(all_data["AQI"]["times"], all_data["AQI"]["values"]):
            parsed = _pt(t) or datetime.now()
            data_dict["datetime"].append(parsed)
            val = v.get("value") if isinstance(v, dict) else v
            data_dict["AQI"].append(val)

    for p in pollutant_names[1:]:
        data_dict[p] = [None] * len(data_dict["datetime"])
        if p in all_data:
            for i, t in enumerate(all_data[p]["times"]):
                if i < len(data_dict["datetime"]):
                    val = all_data[p]["values"][i]
                    data_dict[p][i] = val.get("value") if isinstance(val, dict) else val

    return pd.DataFrame(data_dict)


# ═══════════════════════════════════════════════════════════
#  数据库操作
# ═══════════════════════════════════════════════════════════

def _get_db_engine():
    """获取数据库引擎"""
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
        f"charset={DB_CONFIG['charset']}"
    )


def save_to_database(df: pd.DataFrame, site_code: str) -> bool:
    """将实时数据保存到 MySQL 数据库"""
    try:
        engine = _get_db_engine()
        table_name = f"air_quality_site_{site_code.lower()}"
        df_copy = df.copy()

        if "datetime" in df_copy.columns:
            df_copy["datetime"] = df_copy["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # 确保只保留标准列
        standard_cols = ["datetime", "AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]
        existing_cols = [c for c in standard_cols if c in df_copy.columns]
        df_copy = df_copy[existing_cols]

        # 去重：删除数据库中已存在的时间点
        from sqlalchemy import text

        with engine.connect() as conn:
            existing_times = pd.read_sql(
                text(f"SELECT DISTINCT `datetime` FROM `{table_name}`"), conn
            )
            if len(existing_times) > 0:
                existing_set = set(existing_times["datetime"].astype(str))
                df_copy = df_copy[~df_copy["datetime"].isin(existing_set)]

        if len(df_copy) == 0:
            print(f"  [SKIP] 无新数据需要保存")
            engine.dispose()
            return True

        df_copy.to_sql(name=table_name, con=engine, if_exists="append", index=False)
        print(f"  [OK] 已将 {len(df_copy)} 条新数据保存到 {table_name}")
        engine.dispose()
        return True
    except Exception as e:
        print(f"  [ERROR] 保存数据失败: {e}")
        return False


def check_data_freshness(site_code: str, threshold_hours: int = 2) -> bool:
    """检查数据库中数据的时效性"""
    try:
        engine = _get_db_engine()
        table_name = f"air_quality_site_{site_code.lower()}"
        query = f"SELECT MAX(`datetime`) as latest_time FROM `{table_name}`"
        result = pd.read_sql(query, engine)
        latest_time = result["latest_time"].iloc[0]
        engine.dispose()

        if latest_time is None:
            print(f"  站点 {site_code} 数据库中无数据")
            return False

        if isinstance(latest_time, str):
            latest_time = pd.to_datetime(latest_time)

        current_time = pd.Timestamp.now()
        time_diff = (current_time - latest_time).total_seconds() / 3600

        if time_diff < 0:
            return True
        elif time_diff > threshold_hours:
            print(f"  数据已过时 {time_diff:.1f} 小时，需要更新")
            return False
        else:
            print(f"  [OK] 数据新鲜（{time_diff:.1f} 小时前更新）")
            return True
    except Exception as e:
        print(f"  [FAIL] 检查数据时效性失败: {e}")
        return False


def get_or_fetch_data(
    site_code: str, location: str, force_fetch: bool = False
) -> pd.DataFrame | None:
    """获取数据：优先用数据库，数据不足或过时则实时抓取"""
    if not force_fetch and check_data_freshness(site_code, threshold_hours=2):
        print("  [OK] 数据库数据足够新鲜，直接使用")
        return None

    print("\n  [START] 开始实时获取空气质量数据...")
    df = fetch_realtime_data(site_code, location)

    if df is not None and len(df) > 0:
        save_to_database(df, site_code)
        print(f"  [OK] 实时数据获取并保存成功")
        return df
    else:
        print("  [ERROR] 实时数据获取失败，将使用数据库存量数据")
        return None


# ═══════════════════════════════════════════════════════════
#  批量刷新（供 scripts/refresh_data.py 使用）
# ═══════════════════════════════════════════════════════════

def refresh_all_sites(site_details: dict = None) -> dict:
    """批量刷新所有站点数据

    Returns:
        {"success": int, "failed": int, "details": dict}
    """
    if site_details is None:
        from config import SITE_DETAILS as sd

        site_details = sd

    results = {"success": 0, "failed": 0, "details": {}}

    for site_code, info in site_details.items():
        print(f"\n{'='*50}")
        print(f"[{site_code}] {info['name']} ({info['location']})")
        print(f"{'='*50}")

        location = info["location"]
        df = fetch_realtime_data(site_code, location)

        if df is not None and len(df) > 0:
            ok = save_to_database(df, site_code)
            if ok:
                results["success"] += 1
                results["details"][site_code] = "OK"
            else:
                results["failed"] += 1
                results["details"][site_code] = "SAVE_FAILED"
        else:
            results["failed"] += 1
            results["details"][site_code] = "FETCH_FAILED"

    return results
