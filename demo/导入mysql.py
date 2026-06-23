import pandas as pd
import os
import pymysql
from sqlalchemy import create_engine, text
import sqlalchemy.types
import glob

# 指定站点列表
target_sites = ['2610A', '2611A', '2596A', '2597A', '1916A', '1917A', '3376A', '3377A']

# MySQL数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',
    'database': 'air_data',
    'charset': 'utf8mb4'
}

# 先连接到MySQL（不指定数据库），创建数据库
temp_engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}?"
    f"charset={DB_CONFIG['charset']}"
)

# 创建数据库（如果不存在）
with temp_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    conn.commit()

# 连接到数据库
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
    f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?"
    f"charset={DB_CONFIG['charset']}"
)

# 定义年份范围
years = range(2014, 2026)

# 循环处理每一年
for year in years:
    # 构建数据目录路径（根据实际文件路径调整）

    data_dir = rf'C:\Users\28927\dazuoye\pythonProject3\data\{year}0101-{year}1231'
    
    # 查找该年份下所有CSV文件
    pattern = os.path.join(data_dir, 'china_sites_*.csv')
    csv_files = glob.glob(pattern)
    
    # 如果没有找到文件，尝试包含子目录
    if not csv_files:
        pattern = os.path.join(data_dir, '*', 'china_sites_*.csv')
        csv_files = glob.glob(pattern)
    
    # 循环处理每个CSV文件
    for csv_file in csv_files:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 检查哪些目标站点存在于数据中
        available_sites = [site for site in target_sites if site in df.columns]
        
        if not available_sites:
            continue
        
        # 过滤掉 _24h 和 _8h 指标，只保留原始小时数据
        df_filtered = df[~df['type'].str.contains('_24h|_8h', na=False)]
        
        if df_filtered.empty:
            continue
        
        # 将宽格式转换为长格式
        df_long = pd.melt(
            df_filtered,
            id_vars=['date', 'hour', 'type'],
            value_vars=available_sites,
            var_name='site',
            value_name='value'
        )
        
        # 创建时间列
        df_long['datetime'] = pd.to_datetime(
            df_long['date'].astype(str) + df_long['hour'].astype(str).str.zfill(2),
            format='%Y%m%d%H'
        )
        
        # 为每个站点单独处理数据
        for site in available_sites:
            # 筛选该站点的数据
            site_data = df_long[df_long['site'] == site].copy()
            
            if site_data.empty:
                continue
            
            # 透视表：时间为行，指标为列
            site_pivot = site_data.pivot_table(
                index='datetime',
                columns='type',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # 按时间排序
            site_pivot = site_pivot.sort_values('datetime').reset_index(drop=True)
            
            # 表名（全部转为小写）
            table_name = f'air_quality_site_{site.lower()}'
            
            # 检查表是否存在
            with engine.connect() as conn:
                result = conn.execute(
                    text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = :db AND table_name = :table"),
                    {"db": DB_CONFIG['database'], "table": table_name}
                )
                table_exists = result.scalar() > 0
                
                # 如果表不存在，则创建
                if not table_exists:
                    # 构建创建表的SQL语句（列名用反引号包裹，处理特殊字符）
                    columns_sql = ['`datetime` DATETIME']
                    indicator_cols = [col for col in site_pivot.columns if col != 'datetime']
                    
                    for col in indicator_cols:
                        # 用反引号包裹列名，处理包含小数点等特殊字符的列名
                        columns_sql.append(f'`{col}` FLOAT')
                    
                    create_table_sql = f"""
                    CREATE TABLE `{table_name}` (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        {', '.join(columns_sql)},
                        INDEX idx_datetime (`datetime`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                    """
                    
                    # 执行创建表
                    conn.execute(text(create_table_sql))
                    conn.commit()
            
            # 定义列的数据类型
            column_dtypes = {
                'datetime': sqlalchemy.types.DATETIME
            }
            for col in site_pivot.columns:
                if col != 'datetime':
                    column_dtypes[col] = sqlalchemy.types.FLOAT
            
            # 写入数据库（追加模式）
            site_pivot.to_sql(
                name=table_name,
                con=engine,
                if_exists='append',
                index=False,
                dtype=column_dtypes
            )

# 关闭连接
engine.dispose()
