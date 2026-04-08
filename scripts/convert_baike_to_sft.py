"""
将百度百科QA数据转换为SFT微调格式。

字段映射：
  title  -> instruction
  desc   -> input
  answer -> output

流式逐行处理，内存占用恒定，适用于大文件。

用法：
  python scripts/convert_baike_to_sft.py \
      --input_file dataset/finetuneData/baike_qa2019/baike_qa_train.json \
      --output_file dataset/finetuneData/baike_qa2019/baike_qa_train_sft.jsonl

  # 限制条数（测试用）
  python scripts/convert_baike_to_sft.py \
      --input_file dataset/finetuneData/baike_qa2019/baike_qa_train.json \
      --output_file dataset/finetuneData/baike_qa2019/baike_qa_train_sft.jsonl \
      --max_samples 100
"""

import argparse
import json
import sys
import time


def convert(input_file: str, output_file: str, max_samples: int = 0):
    total = 0
    skipped = 0
    written = 0
    start_time = time.time()

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 10:
                    print(f"跳过无效JSON（第{total}行）: {e}", file=sys.stderr)
                continue

            answer = record.get("answer", "")
            title = record.get("title", "")

            # 跳过 answer 或 title 为空的记录
            if not answer.strip() or not title.strip():
                skipped += 1
                continue

            converted = {
                "instruction": title,
                "input": record.get("desc", ""),
                "output": answer,
            }
            fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
            written += 1

            # 进度显示
            if written % 10000 == 0:
                elapsed = time.time() - start_time
                speed = written / elapsed if elapsed > 0 else 0
                print(f"已处理 {total} 条，写入 {written} 条，"
                      f"跳过 {skipped} 条，速度 {speed:.0f} 条/秒")

            if max_samples > 0 and written >= max_samples:
                break

    elapsed = time.time() - start_time
    print(f"\n转换完成！")
    print(f"  总读取: {total} 条")
    print(f"  成功写入: {written} 条")
    print(f"  跳过: {skipped} 条")
    print(f"  耗时: {elapsed:.1f} 秒")
    print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="将百度百科QA数据转换为SFT微调格式")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入文件路径（JSONL格式，每行一个JSON对象）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径（默认：输入文件名去掉扩展名后加 _sft.jsonl）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="最大转换条数，0表示不限制（用于测试/调试）",
    )
    args = parser.parse_args()

    output_file = args.output_file
    if output_file is None:
        base = args.input_file.rsplit('.', 1)[0]
        output_file = base + "_sft.jsonl"

    print(f"输入文件: {args.input_file}")
    print(f"输出文件: {output_file}")
    if args.max_samples > 0:
        print(f"最大条数: {args.max_samples}")
    print()

    convert(args.input_file, output_file, args.max_samples)


if __name__ == "__main__":
    main()
