"""
一键刷新所有站点数据 — 基于 air.cnemc.cn API 批量获取并保存
使用方式: python scripts/refresh_data.py
"""
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import SITE_DETAILS
from services.data_fetcher import (
    check_data_freshness,
    fetch_realtime_data,
    save_to_database,
)


def refresh_site(site_code: str, location: str) -> bool:
    """刷新单个站点的数据"""
    name = SITE_DETAILS[site_code]["name"]
    print(f"\n[{site_code}] {name} ({location})")
    print("-" * 50)

    # 检查新鲜度
    if check_data_freshness(site_code, threshold_hours=2):
        print("  [SKIP] 数据已是最新")
        return True

    # 尝试获取实时数据
    print("  数据过期或缺失，开始获取...")
    df = fetch_realtime_data(site_code, location)

    if df is not None and len(df) > 0:
        success = save_to_database(df, site_code)
        return success
    else:
        print(f"  [FAIL] 未能获取 {site_code} 的实时数据")
        return False


def main():
    print("=" * 60)
    print("空气质量数据刷新工具")
    print(f"  数据源: air.cnemc.cn API + aqicn.org (备用)")
    print(f"  站点数: {len(SITE_DETAILS)}")
    print("=" * 60)

    results = {"success": 0, "skipped": 0, "failed": 0}

    for i, (code, info) in enumerate(SITE_DETAILS.items(), 1):
        print(f"\n[{i}/{len(SITE_DETAILS)}]", end="")
        t0 = time.time()

        # 先检查新鲜度，已是最新则跳过
        if check_data_freshness(code, threshold_hours=2):
            results["skipped"] += 1
            continue

        ok = refresh_site(code, info["location"])
        elapsed = time.time() - t0

        if ok:
            results["success"] += 1
        else:
            results["failed"] += 1

        print(f"  耗时: {elapsed:.1f}s")

        # 温和限速，避免触发 API 频率限制
        if i < len(SITE_DETAILS):
            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"结果: 成功 {results['success']} | 跳过 {results['skipped']} | 失败 {results['failed']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
