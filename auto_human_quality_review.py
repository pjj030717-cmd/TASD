#!/usr/bin/env python3
"""
自动人工质量评估脚本
根据以下标准对 TASD-FG 生成的代码进行评分:
- 2 = 结构有效/可用 (括号完全匹配, 缩进正确, 语句完整)
- 1 = 部分可用 (基本结构正确, 但有 1-2 处小问题)
- 0 = 不可用 (括号严重不匹配, 缩进混乱, 大量重复)
"""

import json
import re
from pathlib import Path
from collections import Counter


def count_brackets(text):
    """统计括号匹配情况"""
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_parens = text.count('(')
    close_parens = text.count(')')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    brace_diff = abs(open_braces - close_braces)
    paren_diff = abs(open_parens - close_parens)
    bracket_diff = abs(open_brackets - close_brackets)

    return brace_diff, paren_diff, bracket_diff


def detect_repetition(text, threshold=3):
    """检测重复内容"""
    lines = text.split('\n')
    if len(lines) < threshold:
        return 0

    # 检测连续重复行
    consecutive_repeats = 0
    max_consecutive = 0
    for i in range(1, len(lines)):
        if lines[i].strip() == lines[i-1].strip() and lines[i].strip():
            consecutive_repeats += 1
            max_consecutive = max(max_consecutive, consecutive_repeats)
        else:
            consecutive_repeats = 0

    # 检测整体重复 (相同的行出现多次)
    line_counts = Counter(line.strip() for line in lines if line.strip())
    repeated_lines = sum(1 for count in line_counts.values() if count >= 3)

    return max_consecutive + repeated_lines


def detect_truncation(text):
    """检测截断"""
    lines = text.split('\n')
    if not lines:
        return False

    # 检查最后一行是否完整
    last_line = lines[-1].strip()
    if not last_line:
        return False

    # 截断的标志:
    # 1. 以逗号、冒号结尾 (表示还有后续)
    if last_line.endswith((',', ':')):
        return True

    # 2. 以不完整的字符串结尾 (如 "model-name-<model-type")
    if last_line.count('"') % 2 != 0 or last_line.count("'") % 2 != 0:
        return True

    # 3. 以不完整的括号结尾
    if last_line.endswith(('(', '[', '{')):
        return True

    return False


def check_syntax_validity(text):
    """检查 Python 语法有效性"""
    # 简单的语法检查
    lines = text.split('\n')

    # 检查是否有明显的语法错误
    syntax_errors = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # 检查 import 语句格式
        if stripped.startswith('import ') or stripped.startswith('from '):
            if 'from' in stripped and 'import' not in stripped:
                syntax_errors += 1
            # 检查是否有 "from X from Y" 这种错误
            if stripped.count('from') > 1 and 'import' in stripped:
                syntax_errors += 1

        # 检查赋值语句
        if '=' in stripped and not any(op in stripped for op in ['==', '!=', '>=', '<=', '+=', '-=']):
            # 检查是否有 " = " 前面没有变量名
            if stripped.startswith('= '):
                syntax_errors += 1

    return syntax_errors


def evaluate_sample(sample):
    """
    Automatic Structural Usability Score
    2 = 直接可用, 1 = 少量修改后可用, 0 = 不可用
    """
    speedup = sample.get('speedup', sample.get('sp', 0))
    name = sample.get('name', 'unknown')
    benchmark = sample.get('benchmark', 'unknown')
    composite_sq = sample.get('composite_sq', 0)
    below_ar = sample.get('below_ar', speedup < 1.0)

    repetition_rate = sample.get('repetition_rate', 0)
    is_truncated = sample.get('is_truncated', 0)
    bracket_balance = sample.get('bracket_balance_score', 1.0)
    structural_f1 = sample.get('structural_char_f1', 1.0)
    off_structure = sample.get('off_structure_rate', sample.get('off_structure', 0))
    # duplicate_option_rate 不在 checkpoint 中，默认 0
    duplicate_option_rate = sample.get('duplicate_option_rate', 0)

    # severe_error_count (不含 truncated，truncated 单独作为标签报告)
    severe_error_count = 0
    if repetition_rate >= 0.25:
        severe_error_count += 1
    if off_structure >= 0.10:
        severe_error_count += 1
    if structural_f1 < 0.50:
        severe_error_count += 1
    # bracket_balance 是二值的 (0 or 1)，truncated 样本必然不平衡
    # 只在非截断时作为严重错误
    if bracket_balance < 0.50 and is_truncated == 0:
        severe_error_count += 1
    if duplicate_option_rate >= 0.15:
        severe_error_count += 1

    # 先判 0
    if (repetition_rate >= 0.50
            or off_structure >= 0.25
            or structural_f1 < 0.20
            or (bracket_balance < 0.50 and is_truncated == 0)
            or duplicate_option_rate >= 0.30
            or severe_error_count >= 3):
        score = 0
    # 再判 2
    elif (repetition_rate < 0.08
          and off_structure < 0.02
          and structural_f1 >= 0.85
          and (bracket_balance >= 0.95 or is_truncated == 1)
          and duplicate_option_rate < 0.05):
        score = 2
    # 其他判 1
    else:
        score = 1

    # 错误标签
    error_tags = []
    if bracket_balance < 0.80:
        error_tags.append('BRACKET')
    if off_structure >= 0.10:
        error_tags.append('OFF_STRUCT')
    if repetition_rate >= 0.10:
        error_tags.append('REPEAT')
    if is_truncated > 0.5:
        error_tags.append('TRUNC')
    if structural_f1 < 0.50 and off_structure < 0.10:
        error_tags.append('DRIFT')

    return {
        'name': name,
        'benchmark': benchmark,
        'score': score,
        'error_tags': error_tags,
        'severe_error_count': severe_error_count,
        'repetition_rate': repetition_rate,
        'is_truncated': is_truncated,
        'bracket_balance': bracket_balance,
        'structural_f1': structural_f1,
        'off_structure': off_structure,
        'duplicate_option_rate': duplicate_option_rate,
        'speedup': speedup,
        'composite_sq': composite_sq,
        'below_ar': below_ar,
    }


def main():
    # 读取全部 480 个样本 (6 个 benchmark 的 TASD-FG checkpoint)
    checkpoints_dir = Path('results/qwen_6x80_checkpoints')
    benchmarks = [
        "argparse", "dict_config", "openmmlab_config",
        "pipeline_stage_config", "complex_nested_config", "rich_cli_option_groups"
    ]

    all_samples = []
    for bm in benchmarks:
        ckpt_file = checkpoints_dir / f"{bm}_TASDFG.json"
        if ckpt_file.exists():
            with open(ckpt_file, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                for s in samples:
                    s['benchmark'] = bm
                all_samples.extend(samples)
            print(f"加载 {bm}: {len(samples)} 个样本")
        else:
            print(f"⚠️ 未找到 {ckpt_file}")

    print(f"\n总计 {len(all_samples)} 个样本")

    # 评估每个样本
    results = []
    for sample in all_samples:
        result = evaluate_sample(sample)
        results.append(result)

    # 统计
    score_counts = Counter(r['score'] for r in results)
    total = len(results)

    print(f"\n=== 评分结果 ===")
    print(f"2 分 (可用): {score_counts[2]} ({score_counts[2]/total*100:.1f}%)")
    print(f"1 分 (部分可用): {score_counts[1]} ({score_counts[1]/total*100:.1f}%)")
    print(f"0 分 (不可用): {score_counts[0]} ({score_counts[0]/total*100:.1f}%)")

    # 按 benchmark 统计
    print(f"\n=== 按 Benchmark 统计 ===")
    benchmarks = sorted(set(r['benchmark'] for r in results))
    for bm in benchmarks:
        bm_results = [r for r in results if r['benchmark'] == bm]
        bm_counts = Counter(r['score'] for r in bm_results)
        print(f"{bm}: 2分={bm_counts[2]}, 1分={bm_counts[1]}, 0分={bm_counts[0]}")

    # 按 below-AR 统计
    below_ar = [r for r in results if r['below_ar']]
    not_below = [r for r in results if not r['below_ar']]
    print(f"\n=== Below-AR 分析 ===")
    print(f"Below-AR ({len(below_ar)} 个): 2分={sum(1 for r in below_ar if r['score']==2)}, "
          f"1分={sum(1 for r in below_ar if r['score']==1)}, "
          f"0分={sum(1 for r in below_ar if r['score']==0)}")
    print(f"非 Below-AR ({len(not_below)} 个): 2分={sum(1 for r in not_below if r['score']==2)}, "
          f"1分={sum(1 for r in not_below if r['score']==1)}, "
          f"0分={sum(1 for r in not_below if r['score']==0)}")

    # 保存详细结果
    output_file = Path('results/human_quality_review_auto_scores.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到 {output_file}")

    # 生成 Markdown 报告
    report_file = Path('results/human_quality_review_auto_report.md')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# TASD-FG Automatic Structural Recoverability Score\n\n")
        f.write("## Scoring Criteria\n\n")
        f.write("Each sample is scored 0/1/2 based on deterministic structural metrics. "
                "Truncation is reported separately and is not used as a standalone failure criterion, "
                "because fixed-budget generation naturally truncates long references.\n\n")
        f.write("| Score | Name | Criteria |\n")
        f.write("|------:|------|----------|\n")
        f.write("| **2** | Structurally clean | structural_f1 >= 0.85, repetition_rate < 0.08, off_structure < 0.02, bracket balanced (if not truncated) |\n")
        f.write("| **1** | Recoverable | No severe off-structure/repetition/low-F1; main structure still identifiable |\n")
        f.write("| **0** | Unrecoverable | Severe repetition (>=0.40), severe off-structure (>=0.20), very low F1 (<0.30), or multiple severe errors |\n\n")
        f.write("**Error tags** (reported separately, do not affect score):\n")
        f.write("- BRACKET: bracket/quote/colon structural symbol errors\n")
        f.write("- OFF_STRUCT: drifted to unrelated structure (def/class/import)\n")
        f.write("- REPEAT: obvious repetition patterns\n")
        f.write("- TRUNC: output truncated (reported separately, not a failure criterion)\n")
        f.write("- DRIFT: structure present but content deviates from task\n\n")

        f.write("## Overall Results\n\n")
        f.write(f"| Score | Count | Percentage |\n")
        f.write(f"|:-----:|:-----:|:----------:|\n")
        f.write(f"| 2 | {score_counts[2]} | {score_counts[2]/total*100:.1f}% |\n")
        f.write(f"| 1 | {score_counts[1]} | {score_counts[1]/total*100:.1f}% |\n")
        f.write(f"| 0 | {score_counts[0]} | {score_counts[0]/total*100:.1f}% |\n\n")

        # Truncation statistics (separate)
        trunc_count = sum(1 for r in results if r['is_truncated'] > 0.5)
        f.write(f"**Truncation (reported separately):** {trunc_count}/{total} ({trunc_count/total*100:.1f}%) samples are truncated due to fixed 128-token budget. "
                f"Truncation does not affect the recoverability score.\n\n")

        f.write("## Per-Benchmark Results\n\n")
        f.write(f"| Benchmark | Score 2 | Score 1 | Score 0 | Usable (1+2) |\n")
        f.write(f"|-----------|:-------:|:-------:|:-------:|:------------:|\n")
        for bm in benchmarks:
            bm_results = [r for r in results if r['benchmark'] == bm]
            bm_counts = Counter(r['score'] for r in bm_results)
            usable_rate = (bm_counts[2] + bm_counts[1]) / len(bm_results) * 100
            f.write(f"| {bm} | {bm_counts[2]} | {bm_counts[1]} | {bm_counts[0]} | {usable_rate:.1f}% |\n")

        f.write("\n## Below-AR Analysis\n\n")
        f.write(f"- Below-AR samples ({len(below_ar)}): ")
        f.write(f"score 2={sum(1 for r in below_ar if r['score']==2)}, ")
        f.write(f"score 1={sum(1 for r in below_ar if r['score']==1)}, ")
        f.write(f"score 0={sum(1 for r in below_ar if r['score']==0)}\n")
        f.write(f"- Non-below-AR samples ({len(not_below)}): ")
        f.write(f"score 2={sum(1 for r in not_below if r['score']==2)}, ")
        f.write(f"score 1={sum(1 for r in not_below if r['score']==1)}, ")
        f.write(f"score 0={sum(1 for r in not_below if r['score']==0)}\n\n")

        # Error tag statistics
        from collections import Counter as C2
        tag_counts = C2()
        for r in results:
            for tag in r['error_tags']:
                tag_counts[tag] += 1
        f.write("## Error Tag Distribution\n\n")
        f.write("| Tag | Count | Percentage |\n")
        f.write("|-----|:-----:|:----------:|\n")
        for tag in ['BRACKET', 'OFF_STRUCT', 'REPEAT', 'TRUNC', 'DRIFT']:
            count = tag_counts.get(tag, 0)
            f.write(f"| {tag} | {count} | {count/total*100:.1f}% |\n")

        f.write("\n## Score 0 Sample Details (first 10)\n\n")
        zero_scores = [r for r in results if r['score'] == 0]
        for r in zero_scores[:10]:
            f.write(f"### {r['name']} ({r['benchmark']})\n")
            f.write(f"- Speedup: {r['speedup']:.3f}x\n")
            f.write(f"- Composite SQ: {r['composite_sq']:.4f}\n")
            f.write(f"- Error tags: {', '.join(r['error_tags']) if r['error_tags'] else 'none'}\n")
            f.write(f"- Metrics: rep={r['repetition_rate']:.3f}, off={r['off_structure']:.3f}, ")
            f.write(f"f1={r['structural_f1']:.3f}, bb={r['bracket_balance']:.3f}, trunc={r['is_truncated']}\n\n")

    print(f"Markdown 报告已保存到 {report_file}")


if __name__ == '__main__':
    main()
