#!/usr/bin/env python
"""
网络爬虫命令行入口

根据配置文件批量爬取网站内容，或查看爬虫状态。

Usage:
    # 从配置文件批量爬取
    python scripts/crawl.py run --config configs/crawler/crawler_config.yaml --parallel 2

    # 查看爬虫状态
    python scripts/crawl.py status

    # 指定输出目录
    python scripts/crawl.py run --config configs/crawler/crawler_config.yaml --output-dir ./output/crawled
"""
import os
import sys
import asyncio
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_run(args):
    """执行批量爬取"""
    from src.crawler.engine import crawl_sites

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print("Crawler - Batch Mode")
    print("=" * 60)
    print(f"Config:      {config_path}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Parallel:    {args.parallel}")
    print("=" * 60)

    total_pages = asyncio.run(crawl_sites(
        config_path=config_path,
        output_dir=args.output_dir,
        max_concurrent=args.parallel,
    ))

    if total_pages > 0:
        print(f"\nDone! Total pages crawled: {total_pages}")
    else:
        print("\nNo pages were crawled. Check your config file.")
    return total_pages


def cmd_status(args):
    """显示爬虫状态"""
    from src.crawler.storage.database import get_crawl_stats_repo, get_crawler_repo
    from src.utils.database import DatabaseManager

    db_path = args.db
    if db_path and not os.path.exists(db_path):
        print(f"数据库不存在: {db_path}")
        sys.exit(1)

    try:
        page_repo = get_crawler_repo(db_path)
        stats_repo = get_crawl_stats_repo(db_path)
    except Exception as e:
        print(f"无法连接数据库: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Crawler Status")
    print("=" * 60)
    print(f"Database: {db_path or 'default (db/transform.db)'}")
    print("=" * 60)

    # 页面统计
    crawled_count = page_repo.get_crawled_count()
    print(f"Crawled pages:  {crawled_count}")

    # 最近统计
    recent = stats_repo.get_recent_stats(days=args.days)
    if recent:
        print(f"\nRecent {len(recent)} days stats:")
        print("-" * 60)
        for record in recent:
            date = record.get("date", "N/A")
            pages = record.get("pages_crawled", 0)
            failed = record.get("pages_failed", 0)
            queued = record.get("pages_queued", 0)
            total_time = record.get("total_time", 0.0)
            print(f"  {date}  pages={pages}  failed={failed}  queued={queued}  time={total_time:.1f}s")
    else:
        print("No crawl stats available.")

    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="网络爬虫命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 从配置文件批量爬取
  python scripts/crawl.py run --config configs/crawler/crawler_config.yaml --parallel 2

  # 查看爬虫状态（最近7天）
  python scripts/crawl.py status

  # 查看最近30天的统计
  python scripts/crawl.py status --days 30
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # --- run 子命令 ---
    run_parser = subparsers.add_parser("run", help="从配置文件批量爬取")
    run_parser.add_argument(
        "--config", type=str,
        default="configs/crawler/crawler_config.yaml",
        help="爬虫配置文件路径（默认 configs/crawler/crawler_config.yaml）",
    )
    run_parser.add_argument(
        "--output-dir", type=str,
        default="./output/crawled",
        help="输出目录（默认 ./output/crawled）",
    )
    run_parser.add_argument(
        "--parallel", type=int,
        default=2,
        help="并行站点数（默认 2）",
    )

    # --- status 子命令 ---
    status_parser = subparsers.add_parser("status", help="显示爬虫状态")
    status_parser.add_argument(
        "--db", type=str, default=None,
        help="数据库路径（默认 db/transform.db）",
    )
    status_parser.add_argument(
        "--days", type=int, default=7,
        help="显示最近 N 天的统计（默认 7）",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    return args


def main():
    args = parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
