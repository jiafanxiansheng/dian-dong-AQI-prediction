"""
数据导入脚本 — 将 data/ 目录下的 CSV 文件导入 MySQL 数据库
使用方式: python scripts/import_data.py
"""
import os
import sys
import glob
import pandas as pd
import sqlalchemy.types
from sqlalchemy import create_engine, text

# 确保能导入项目配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import DB_CONFIG, TARGET_SITES


def create_database():
    """创建数据库（如果不存在）"""
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}?"
        f"charset={DB_CONFIG['charset']}"
    )
    with engine.connect() as conn:
        conn.execute(text(
            f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        ))
        conn.commit()
    engine.dispose()
    print(f"✓ 数据库 {DB_CONFIG['database']} 已就绪")


def get_engine():
    """获取数据库引擎"""
    return create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
        f"charset={DB_CONFIG['charset']}"
    )


def import_year_data(year: int, engine):
    """导入指定年份的所有 CSV 数据"""
    data_dir = os.path.join(BASE_DIR, "data", f"{year}0101-{year}1231")
    if not os.path.exists(data_dir):
        print(f"  ⚠️ 数据目录不存在: {data_dir}")
        return 0

    # 查找 CSV 文件
    pattern = os.path.join(data_dir, "china_sites_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        # 尝试子目录
        pattern = os.path.join(data_dir, "*", "china_sites_*.csv")
        csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"  ⚠️ {year} 年未找到 CSV 文件")
        return 0

    imported_count = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            # 只保留目标站点
            available_sites = [s for s in TARGET_SITES if s in df.columns]
            if not available_sites:
                continue

            # 过滤 24h/8h 聚合指标，只保留小时数据
            df_filtered = df[~df["type"].str.contains("_24h|_8h", na=False)]
            if df_filtered.empty:
                continue

            # 宽表转长表
            df_long = pd.melt(
                df_filtered,
                id_vars=["date", "hour", "type"],
                value_vars=available_sites,
                var_name="site",
                value_name="value",
            )

            # 创建时间列
            df_long["datetime"] = pd.to_datetime(
                df_long["date"].astype(str) + df_long["hour"].astype(str).str.zfill(2),
                format="%Y%m%d%H",
            )

            # 每个站点一张表
            for site in available_sites:
                site_data = df_long[df_long["site"] == site].copy()
                if site_data.empty:
                    continue

                site_pivot = site_data.pivot_table(
                    index="datetime", columns="type", values="value", aggfunc="first"
                ).reset_index()
                site_pivot = site_pivot.sort_values("datetime").reset_index(drop=True)

                table_name = f"air_quality_site_{site.lower()}"

                # 首次创建表
                with engine.connect() as conn:
                    result = conn.execute(
                        text(
                            "SELECT COUNT(*) FROM information_schema.tables "
                            "WHERE table_schema = :db AND table_name = :tbl"
                        ),
                        {"db": DB_CONFIG["database"], "tbl": table_name},
                    )
                    table_exists = result.scalar() > 0

                    if not table_exists:
                        columns_sql = ["`datetime` DATETIME"]
                        for col in site_pivot.columns:
                            if col != "datetime":
                                columns_sql.append(f"`{col}` FLOAT")
                        create_sql = f"""
                        CREATE TABLE `{table_name}` (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            {', '.join(columns_sql)},
                            INDEX idx_datetime (`datetime`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                        """
                        conn.execute(text(create_sql))
                        conn.commit()

                # 写入数据
                col_dtypes = {"datetime": sqlalchemy.types.DATETIME}
                for col in site_pivot.columns:
                    if col != "datetime":
                        col_dtypes[col] = sqlalchemy.types.FLOAT

                site_pivot.to_sql(
                    name=table_name, con=engine,
                    if_exists="append", index=False,
                    dtype=col_dtypes,
                )

            imported_count += len(csv_files)
        except Exception as e:
            print(f"  ✗ 处理 {os.path.basename(csv_file)} 失败: {e}")

    return imported_count


if __name__ == "__main__":
    print("=" * 60)
    print("📥 滇东AQI数据导入工具")
    print("=" * 60)

    create_database()
    engine = get_engine()

    years = range(2014, 2026)
    total_files = 0

    for year in years:
        print(f"\n📂 处理 {year} 年数据...")
        count = import_year_data(year, engine)
        total_files += count
        print(f"  ✓ {year} 年完成，处理 {count} 个文件")

    engine.dispose()
    print(f"\n✅ 全部完成！共处理 {total_files} 个文件")
