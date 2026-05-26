from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
import warnings
import re

warnings.filterwarnings('ignore')

# ==================== 数据库配置 ====================
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

# 站点映射（地区 -> 网站上的站点名称）
LOCATION_SITE_MAP = {
    '文山州': '文山',
    '昭通': '昭通市',
    '曲靖': '曲靖市'
}

# 省份ID映射（根据网站实际ID调整）
PROVINCE_ID_MAP = {
    '云南省': 25  # 第25个省份
}


def init_driver(headless=True):
    """初始化浏览器驱动"""
    options = Options()

    if headless:
        options.add_argument("--headless")  # 无头模式，不显示浏览器

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    driver = webdriver.Edge(options=options)
    driver.set_page_load_timeout(30)
    driver.set_script_timeout(30)

    return driver


def parse_chinese_datetime(time_str):
    """
    解析中文时间格式，如 '00时', '01时' 等
    返回完整的datetime对象
    
    智能判断日期：
    - 如果小时数 >= 当前小时，使用今天
    - 如果小时数 < 当前小时，且差距较大，可能是明天的数据
    """
    try:
        # 提取小时数
        match = re.search(r'(\d+)时', time_str)
        if match:
            hour = int(match.group(1))
            now = datetime.now()
            current_hour = now.hour
            
            # 智能判断日期
            if hour >= current_hour:
                # 小时数 >= 当前小时，使用今天
                return now.replace(hour=hour, minute=0, second=0, microsecond=0)
            else:
                # 小时数 < 当前小时
                # 如果差距较大（如当前23点，数据02点），可能是明天
                # 如果差距较小（如当前14点，数据02点），可能是今天凌晨
                hour_diff = current_hour - hour
                if hour_diff > 12:
                    # 差距超过12小时，认为是明天的数据
                    from datetime import timedelta
                    tomorrow = now + timedelta(days=1)
                    return tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
                else:
                    # 差距较小，认为是今天的数据
                    return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        return None
    except:
        return None


def fetch_realtime_data(location, headless=True):
    """
    从网站获取实时空气质量数据

    Args:
        location: 地区名称（如'曲靖'、'文山州'、'昭通'）
        headless: 是否使用无头模式（不显示浏览器）

    Returns:
        dict: 包含污染物数据的字典，失败返回None
    """
    driver = None
    try:
        print(f"🌐 正在获取 {location} 的实时空气质量数据...")

        # 初始化浏览器
        driver = init_driver(headless=headless)
        driver.get("https://air.cnemc.cn:18007/")
        time.sleep(5)

        # 点击"点位空气质量"
        search_box = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.LINK_TEXT, '点位空气质量'))
        )
        search_box.click()
        time.sleep(3)

        # 选择云南省（第25个省份）
        province = WebDriverWait(driver, 15).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="province-list"]/li[25]'))
        )
        province.click()
        time.sleep(3)

        # 选择城市 - 使用更宽松的匹配方式
        city_name = LOCATION_SITE_MAP.get(location, location)
        print(f"  正在查找城市: {city_name}")

        # 尝试多种XPath方式来查找城市
        city = None
        xpath_options = [
            f'//li[contains(text(),"{city_name}")]',
            f'//a[contains(text(),"{city_name}")]',
            f'//div[contains(text(),"{city_name}")]'
        ]

        for xpath in xpath_options:
            try:
                city = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                print(f"  ✓ 找到城市元素 (XPath: {xpath})")
                break
            except:
                continue

        if city is None:
            # 如果都没找到，打印页面结构帮助调试
            print(f"  ⚠️ 未找到城市 {city_name}，尝试列出所有可用城市...")
            try:
                all_cities = driver.find_elements(By.XPATH, '//li[contains(@class, "city")] | //ul[@id="city-list"]/li')
                print(f"  可用的城市数量: {len(all_cities)}")
                for i, c in enumerate(all_cities[:10]):
                    print(f"    {i+1}. {c.text}")
            except:
                pass
            raise Exception(f"无法找到城市: {city_name}")

        city.click()
        time.sleep(3)

        # 选择具体站点 - 如果是曲靖，选择"曲靖师范学院"
        if location == '曲靖':
            station_name = '曲靖师范学院'
        else:
            station_name = city_name.replace('市', '')

        print(f"  正在查找站点: {station_name}")

        station = None
        station_xpath_options = [
            f'//div[contains(text(),"{station_name}")]',
            f'//li[contains(text(),"{station_name}")]',
            f'//a[contains(text(),"{station_name}")]'
        ]

        for xpath in station_xpath_options:
            try:
                station = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                print(f"  ✓ 找到站点元素 (XPath: {xpath})")
                break
            except:
                continue

        if station is None:
            raise Exception(f"无法找到站点: {station_name}")

        station.click()
        time.sleep(3)

        # 获取各污染物数据（包括AQI）
        # 根据HTML截图，AQI的_level是"Quality"而不是"Level"
        pollutants = ['Quality', 'PM2_5Level', 'PM10Level', 'SO2Level', 'NO2Level', 'COLevel', 'O3Level']
        pollutant_names = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

        all_data = {}

        for level, name in zip(pollutants, pollutant_names):
            try:
                # 点击污染物选项
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, f'//div[@_level="{level}"]'))
                )
                element.click()
                time.sleep(1)

                # 执行JS获取图表数据
                chart_data = driver.execute_script("""
                    var myChart = echarts.getInstanceByDom(document.getElementById('pollutantChart'));
                    if (!myChart) return null;
                    var option = myChart.getOption();
                    return {
                        times: option.xAxis[0].data,
                        values: option.series[0].data
                    };
                """)

                if chart_data and chart_data['times'] and chart_data['values']:
                    all_data[name] = {
                        'times': chart_data['times'],
                        'values': chart_data['values']
                    }
                    print(f"  ✓ 获取到 {name} 数据: {len(chart_data['times'])} 条")
                else:
                    print(f"  ⚠️ {name} 数据为空")

            except Exception as e:
                print(f"  ✗ 获取 {name} 数据失败: {e}")
                continue

        if not all_data:
            print("❌ 未能获取任何数据")
            return None

        # 整合数据为DataFrame格式
        df = integrate_data(all_data, location)

        print(f"✅ 成功获取 {location} 的实时数据，共 {len(df)} 条记录")
        return df

    except Exception as e:
        print(f"❌ 获取实时数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def integrate_data(all_data, location):
    """
    整合各污染物数据为统一的DataFrame

    Args:
        all_data: 各污染物数据字典
        location: 地区名称

    Returns:
        DataFrame: 整合后的数据
    """
    # 找到所有污染物的共同时间点
    all_times = set()
    for pollutant, data in all_data.items():
        all_times.update(data['times'])

    all_times = sorted(list(all_times))

    # 创建数据字典（包含AQI）
    data_dict = {
        'datetime': [],
        'AQI': [],
        'PM2.5': [],
        'PM10': [],
        'SO2': [],
        'NO2': [],
        'CO': [],
        'O3': []
    }

    for t in all_times:
        # 解析中文时间格式
        parsed_time = parse_chinese_datetime(t)
        if parsed_time is None:
            parsed_time = datetime.now()  # 如果解析失败，使用当前时间

        data_dict['datetime'].append(parsed_time)
        for pollutant in ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
            if pollutant in all_data:
                times = all_data[pollutant]['times']
                values = all_data[pollutant]['values']
                if t in times:
                    idx = times.index(t)
                    value = values[idx]
                    # 如果值是字典，提取'value'字段的数值
                    if isinstance(value, dict):
                        value = value.get('value', None)
                    data_dict[pollutant].append(value)
                else:
                    data_dict[pollutant].append(None)
            else:
                data_dict[pollutant].append(None)

    df = pd.DataFrame(data_dict)
    # datetime已经是datetime对象，不需要再转换
    # df['datetime'] = pd.to_datetime(df['datetime'])

    return df


def save_to_database(df, site_code):
    """
    将实时数据保存到MySQL数据库

    Args:
        df: 数据DataFrame
        site_code: 站点代码（如'1916A'）

    Returns:
        bool: 是否保存成功
    """
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
            f"charset={DB_CONFIG['charset']}"
        )

        table_name = f'air_quality_site_{site_code.lower()}'

        # 创建副本以避免修改原始数据
        df_copy = df.copy()

        # 将datetime列转换为字符串格式（YYYY-MM-DD HH:MM:SS）
        if 'datetime' in df_copy.columns:
            df_copy['datetime'] = df_copy['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 将污染物列的 'NA' 字符串转换为 None（数据库中的NULL）
        pollutant_cols = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
        for col in pollutant_cols:
            if col in df_copy.columns:
                # 将 'NA' 替换为 None
                df_copy[col] = df_copy[col].replace('NA', None)
                # 将字符串类型的数字转换为float
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # 重新排序列以匹配数据库表结构
        columns_order = ['datetime', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
        existing_cols = [col for col in columns_order if col in df_copy.columns]
        df_copy = df_copy[existing_cols]

        # 追加数据到数据库（不使用method='multi'，改用默认方式）
        df_copy.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',
            index=False
        )

        print(f"✓ 已将 {len(df_copy)} 条数据保存到数据库表 {table_name}")
        return True

    except Exception as e:
        print(f" 保存数据到数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_freshness(site_code, threshold_hours=2):
    """
    检查数据库中数据的时效性

    Args:
        site_code: 站点代码
        threshold_hours: 数据新鲜度阈值（小时）

    Returns:
        bool: 数据是否足够新鲜
    """
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
            f"charset={DB_CONFIG['charset']}"
        )

        table_name = f'air_quality_site_{site_code.lower()}'
        query = f"SELECT MAX(`datetime`) as latest_time FROM `{table_name}`"

        result = pd.read_sql(query, engine)
        latest_time = result['latest_time'].iloc[0]

        if latest_time is None:
            print(f"  站点 {site_code} 数据库中无数据")
            return False

        # 确保latest_time是datetime对象
        if isinstance(latest_time, str):
            latest_time = pd.to_datetime(latest_time)
        
        # 使用pandas的Timestamp确保时区一致
        current_time = pd.Timestamp.now()
        
        # 计算时间差（小时）
        time_diff = (current_time - latest_time).total_seconds() / 3600

        if time_diff < 0:
            # 如果数据时间比当前时间还新，说明是预测数据或时间戳有问题
            print(f"  数据时间为 {latest_time}，使用数据库最新数据")
            return True
        elif time_diff > threshold_hours:
            print(f"  数据已过时 {time_diff:.1f} 小时，需要更新")
            return False
        else:
            print(f"✓ 数据新鲜（{time_diff:.1f} 小时前更新）")
            return True

    except Exception as e:
        print(f"✗ 检查数据时效性失败: {e}")
        return False


def get_or_fetch_data(site_code, location, force_fetch=False):
    """
    获取数据：优先使用数据库，如果数据不足则实时抓取

    Args:
        site_code: 站点代码
        location: 地区名称
        force_fetch: 是否强制实时抓取

    Returns:
        DataFrame or None
    """
    # 检查数据时效性
    if not force_fetch and check_data_freshness(site_code, threshold_hours=2):
        print("✓ 使用数据库中的最新数据")
        return None  # 返回None表示使用数据库数据

    # 需要实时抓取
    print("\n🔄 开始实时获取空气质量数据...")
    df = fetch_realtime_data(location, headless=True)

    if df is not None:
        # 保存到数据库
        save_to_database(df, site_code)
        print("✅ 实时数据获取并保存成功")
        return df
    else:
        print("❌ 实时数据获取失败")
        return None


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🌍 实时空气质量数据获取系统")
    print("=" * 60)

    # 测试获取曲靖的数据
    test_location = '文山'
    test_site_code = '2610A'

    df = get_or_fetch_data(test_site_code, test_location, force_fetch=True)

