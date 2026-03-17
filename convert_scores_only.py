"""
将parquet文件中的scores格式从 "move score move score ..." 转换为 "score score ..."

原格式:
<think>e2e4 85 d2d4 78 g1f3 72 ...</think>

新格式:
<think>85 78 72 ...</think>

同时在user prompt最后添加指令。

因为走法顺序与user prompt中的legal_moves一致，所以不需要重复输出走法名称。

Usage:
    python convert_scores_only.py \
        --input ./data/ChessExplained_scored_obs_adaptk.parquet \
        --output ./data/ChessExplained_scored_obs_adaptk_scores_only.parquet
"""

import re
import argparse
import pandas as pd
from tqdm import tqdm

USER_INSTRUCTION = "Evaluate all legal moves and select the best one."


def extract_scores_only(think_content):
    """
    从 "move1 score1 move2 score2 ..." 格式中提取分数

    输入: "e2e4 85 d2d4 78 g1f3 72"
    输出: "85 78 72"
    """
    parts = think_content.strip().split()
    # 每两个元素一组: (move, score)，只保留score
    scores = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            scores.append(parts[i + 1])
    return " ".join(scores)


def convert_text(text):
    """
    转换单条text:
    1. 将<think>中的内容改为只有分数
    2. 在user prompt最后添加USER_INSTRUCTION
    """
    # 1. 匹配 <think>...</think>，转换为只有分数
    pattern = r'<think>(.*?)</think>'

    def replace_think(match):
        think_content = match.group(1)
        scores_only = extract_scores_only(think_content)
        return f'<think>{scores_only}</think>'

    new_text = re.sub(pattern, replace_think, text, flags=re.DOTALL)

    # 2. 在user prompt最后添加指令（在</chess_position>后，<|im_end|>前）
    # 先检查是否已经有指令，避免重复添加
    if USER_INSTRUCTION not in new_text:
        new_text = new_text.replace(
            "</chess_position><|im_end|>",
            f"</chess_position>\n{USER_INSTRUCTION}<|im_end|>"
        )

    return new_text


def main():
    parser = argparse.ArgumentParser(
        description="Convert scores format from 'move score' to 'score' only"
    )
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    args = parser.parse_args()

    print(f"Loading {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Rows: {len(df)}")

    # 转换
    print("Converting...")
    tqdm.pandas(desc="Processing")
    df['text'] = df['text'].progress_apply(convert_text)

    # 保存
    print(f"Saving to {args.output}")
    df.to_parquet(args.output, index=False)

    # 显示样例
    if len(df) > 0:
        print("\n" + "="*60)
        print("Sample (before -> after):")
        print(df['text'].iloc[0])
        print("="*60)


if __name__ == "__main__":
    main()
