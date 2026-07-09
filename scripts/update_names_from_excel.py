"""
update_names_from_excel.py

从 Excel 文件的 faceInfo sheet 中读取「文件名」和「姓名」，
按文件名匹配数据库 face_records 表的 file_path（忽略路径），
更新匹配记录的 name 字段。

独立运行，不依赖项目中的其他模块。

用法:
    uv run python scripts/update_names_from_excel.py [--dry-run]
"""

import os
import sys
import argparse
from pathlib import Path

import xlrd
import psycopg2

# ==========================================
# 配置
# ==========================================
EXCEL_FILES = [
    "/data/wcm/libface_ljyr.xls",
    "/data/wcm/libface_szmg.xls",
    "/data/wcm/libface_lmgy.xls",
]

DB_HOST = os.getenv("WCM_DB_HOST", "localhost")
DB_PORT = int(os.getenv("WCM_DB_PORT", "5433"))
DB_NAME = os.getenv("WCM_DB_NAME", "facerec")
DB_USER = os.getenv("WCM_DB_USER", "postgres")
DB_PASSWORD = os.getenv("WCM_DB_PASSWORD", "postgres")

SHEET_NAME = "faceInfo"
COL_FILENAME = "文件名"
COL_NAME = "姓名"


def load_excel_mapping(excel_files: list[str]) -> dict[str, str]:
    """从多个 Excel 文件中读取 文件名 -> 姓名 的映射。"""
    mapping: dict[str, str] = {}
    for filepath in excel_files:
        if not os.path.exists(filepath):
            print(f"[WARN] Excel 文件不存在，跳过: {filepath}")
            continue

        wb = xlrd.open_workbook(filepath)
        if SHEET_NAME not in wb.sheet_names():
            print(f"[WARN] 未找到 sheet '{SHEET_NAME}'，跳过: {filepath}")
            continue

        sh = wb.sheet_by_name(SHEET_NAME)

        # 找到列索引
        header = [sh.cell_value(0, c) for c in range(sh.ncols)]
        try:
            filename_col = header.index(COL_FILENAME)
            name_col = header.index(COL_NAME)
        except ValueError as e:
            print(f"[WARN] 缺少必需列 ({e})，跳过: {filepath}")
            continue

        count = 0
        for row in range(1, sh.nrows):
            filename = str(sh.cell_value(row, filename_col)).strip()
            name = str(sh.cell_value(row, name_col)).strip()
            if filename and name:
                mapping[filename] = name
                count += 1

        print(f"[INFO] 从 {Path(filepath).name} 读取了 {count} 条映射")

    return mapping


def update_database(mapping: dict[str, str], dry_run: bool = False):
    """将映射更新到数据库。"""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    conn.autocommit = False
    cur = conn.cursor()

    # 查询所有 face_records 的 id, name, file_path
    cur.execute("SELECT id, name, file_path FROM face_records WHERE file_path IS NOT NULL")
    rows = cur.fetchall()
    print(f"[INFO] 数据库中共 {len(rows)} 条 face_records（有 file_path）")

    updated = 0
    skipped_same = 0
    not_matched = 0

    for record_id, current_name, file_path in rows:
        # 提取文件名（忽略路径）
        db_filename = os.path.basename(file_path)

        if db_filename not in mapping:
            not_matched += 1
            continue

        new_name = mapping[db_filename]
        if new_name == current_name:
            skipped_same += 1
            continue

        if dry_run:
            print(f"  [DRY-RUN] {db_filename}: '{current_name}' -> '{new_name}'")
        else:
            cur.execute(
                "UPDATE face_records SET name = %s WHERE id = %s",
                (new_name, record_id),
            )
        updated += 1

    deleted = 0
    # 清理未匹配到且名字为纯数字序号的记录
    cur.execute("SELECT COUNT(*) FROM face_records WHERE name ~ '^[0-9]+$'")
    numeric_count = cur.fetchone()[0]
    
    if numeric_count > 0:
        if dry_run:
            print(f"  [DRY-RUN] 将删除 {numeric_count} 条未匹配到的纯数字序号记录")
            deleted = numeric_count
        else:
            cur.execute("DELETE FROM face_records WHERE name ~ '^[0-9]+$'")
            deleted = cur.rowcount
            print(f"[INFO] 删除了 {deleted} 条未匹配的纯数字序号记录")

    if not dry_run and (updated > 0 or deleted > 0):
        conn.commit()
        print(f"[INFO] 已提交事务")
    elif dry_run:
        conn.rollback()

    cur.close()
    conn.close()

    print("-" * 40)
    print(f"匹配并更新: {updated}")
    print(f"匹配但名字相同（跳过）: {skipped_same}")
    print(f"未匹配到 Excel: {not_matched}")
    print(f"清理未匹配的数字名记录: {deleted}")
    print(f"总计处理 (查询到的 file_path): {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="从 Excel 更新 face_records 的 name 字段")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览变更，不实际写入数据库",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 40)
        print("  DRY-RUN 模式 — 不会修改数据库")
        print("=" * 40)

    # 1. 读取 Excel 映射
    mapping = load_excel_mapping(EXCEL_FILES)
    print(f"[INFO] 共加载 {len(mapping)} 条文件名->姓名映射")

    if not mapping:
        print("[ERROR] 没有读取到任何映射，退出")
        sys.exit(1)

    # 2. 更新数据库
    update_database(mapping, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
