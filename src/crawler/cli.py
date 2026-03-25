# -*- coding: utf-8 -*-
"""
爬虫命令行接口
Crawler Command Line Interface
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# 设置控制台为UTF-8编码（Windows）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.crawler.engine import crawl_sites
from src.crawler.storage.database import create_sqlite_db
from src.utils.logging import setup_logger


DEFAULT_CONFIG = "configs/crawler/crawler_config.yaml"


def status_command(output_dir: str = "./output/crawled") -> bool:
    """显示爬虫状态"""
    db_path = os.path.join(output_dir, "crawler.db")

    if not os.path.exists(db_path):
        print("No crawl data found. Run a crawl first.")
        return False

    database = create_sqlite_db(db_path)
    crawled_count = database.get_crawled_count()
    stats = database.get_stats(days=1)

    print("\n=== Crawler Status ===")
    print(f"\nPages crawled: {crawled_count}")

    if stats:
        print(f"\nToday's Stats:")
        for s in stats:
            print(f"  Pages crawled: {s['pages_crawled']}")
            print(f"  Pages failed: {s['pages_failed']}")
            print(f"  Bytes downloaded: {s['bytes_downloaded'] / 1024 / 1024:.2f} MB")
            print(f"  Time: {s['total_time']:.1f}s")

    database.close()
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="爬虫批量抓取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # run 命令
    run_parser = subparsers.add_parser('run', help='运行批量爬取')
    run_parser.add_argument('--config', default=DEFAULT_CONFIG, help='配置文件路径')
    run_parser.add_argument('--output-dir', default='./output/crawled', help='输出目录')
    run_parser.add_argument('--parallel', type=int, default=2, help='并行数')
    run_parser.add_argument('--log-dir', default='./logs', help='日志目录')

    # status 命令
    status_parser = subparsers.add_parser('status', help='查看爬虫状态')
    status_parser.add_argument('--output-dir', default='./output/crawled', help='输出目录')

    args = parser.parse_args()

    if args.command == 'run':
        success = asyncio.run(crawl_sites(
            config_path=args.config,
            output_dir=args.output_dir,
            max_concurrent=args.parallel,
        ))
        sys.exit(0 if success > 0 else 1)

    elif args.command == 'status':
        success = status_command(output_dir=args.output_dir)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()