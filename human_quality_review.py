#!/usr/bin/env python3
"""
人工质量评估脚本
从 TASD-FG 结果中抽取样本，生成人工评估表单
"""

import json
import random
from pathlib import Path

def load_json(path):
    """加载 JSON 文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"无法加载 {path}: {e}")
        return None

def sample_tasd_fg_results(n_samples=60):
    """从 TASD-FG checkpoint 中抽样"""
    print("=" * 70)
    print("人工质量评估样本抽取")
    print("=" * 70)
    
    results_dir = Path("/root/autodl-tmp/results")
    checkpoints_dir = results_dir / "qwen_6x80_checkpoints"
    data_dir = Path("/root/autodl-tmp/data")
    
    if not checkpoints_dir.exists():
        print(f"❌ 未找到 {checkpoints_dir}")
        return None
    
    # 收集所有 TASD-FG 样本
    all_samples = []
    
    benchmarks = [
        ("argparse", "codesearchnet_argparse_blocks_80.jsonl"),
        ("dict_config", "codesearchnet_dict_config_blocks_80.jsonl"),
        ("openmmlab_config", "ml_config_blocks_openmmlab_80.jsonl"),
        ("pipeline_stage_config", "pipeline_stage_config_80.jsonl"),
        ("complex_nested_config", "complex_nested_config_80.jsonl"),
        ("rich_cli_option_groups", "rich_cli_option_groups_80.jsonl"),
    ]
    
    for benchmark, data_filename in benchmarks:
        # 读取 checkpoint
        ckpt_file = checkpoints_dir / f"{benchmark}_TASDFG.json"
        if not ckpt_file.exists():
            print(f"  ⚠️  未找到 {ckpt_file.name}")
            continue
        
        with open(ckpt_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        # 读取原始数据以获取 prompt 和 reference
        data_file = data_dir / data_filename
        if not data_file.exists():
            print(f"  ⚠️  未找到 {data_file.name}")
            continue
        
        original_data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                original_data[item['name']] = item
        
        # 合并数据
        for sample in samples:
            name = sample.get('name', '')
            orig = original_data.get(name, {})
            
            all_samples.append({
                'benchmark': benchmark,
                'name': name,
                'prompt': orig.get('prompt', ''),
                'reference': orig.get('reference', ''),
                'generated': sample.get('text', ''),
                'speedup': sample.get('sp', 0),
                'composite_sq': sample.get('composite_sq', 0),
                'off_structure': sample.get('off_structure_rate', 0),
                'below_ar': sample.get('sp', 0) < 1.0
            })
    
    print(f"\n✓ 共找到 {len(all_samples)} 个 TASD-FG 样本")
    
    # 分层抽样
    # 1. 所有 below-AR 样本
    below_samples = [s for s in all_samples if s['below_ar']]
    print(f"  - Below-AR 样本: {len(below_samples)}")
    
    # 2. 低 composite_sq 样本 (bottom 10%)
    sorted_by_sq = sorted(all_samples, key=lambda x: x['composite_sq'])
    low_sq_count = max(5, int(len(all_samples) * 0.1))
    low_sq_samples = sorted_by_sq[:low_sq_count]
    print(f"  - 低 SQ 样本: {len(low_sq_samples)}")
    
    # 3. 高 off_structure 样本 (top 10%)
    sorted_by_off = sorted(all_samples, key=lambda x: x['off_structure'], reverse=True)
    high_off_count = max(5, int(len(all_samples) * 0.1))
    high_off_samples = sorted_by_off[:high_off_count]
    print(f"  - 高 off_structure 样本: {len(high_off_samples)}")
    
    # 4. 随机正常样本
    normal_samples = [s for s in all_samples if not s['below_ar'] and s['composite_sq'] > 0.6 and s['off_structure'] < 0.1]
    random_count = max(0, n_samples - len(below_samples) - len(low_sq_samples) - len(high_off_samples))
    random_samples = random.sample(normal_samples, min(random_count, len(normal_samples)))
    print(f"  - 随机正常样本: {len(random_samples)}")
    
    # 合并并去重
    selected = []
    seen = set()
    
    for sample in below_samples + low_sq_samples + high_off_samples + random_samples:
        key = (sample['benchmark'], sample['name'])
        if key not in seen:
            selected.append(sample)
            seen.add(key)
    
    # 如果不够，补充随机样本
    if len(selected) < n_samples:
        remaining = [s for s in all_samples if (s['benchmark'], s['name']) not in seen]
        additional = random.sample(remaining, min(n_samples - len(selected), len(remaining)))
        selected.extend(additional)
    
    print(f"\n✓ 最终抽取 {len(selected)} 个样本")
    
    # 统计分布
    benchmarks = {}
    for s in selected:
        benchmarks[s['benchmark']] = benchmarks.get(s['benchmark'], 0) + 1
    
    print("\n样本分布:")
    for bench, count in sorted(benchmarks.items()):
        print(f"  - {bench}: {count}")
    
    return selected

def generate_review_form(samples, output_path):
    """生成人工评估表单"""
    print(f"\n📝 生成评估表单: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 人工质量评估表单\n\n")
        f.write("## 评估说明\n\n")
        f.write("请对每个样本的生成结果进行评分：\n\n")
        f.write("- **0 = 不可用**: 结构完全错误，无法使用\n")
        f.write("- **1 = 部分可用**: 结构基本正确，但有明显问题需要修改\n")
        f.write("- **2 = 结构有效/可用**: 结构完整正确，可直接使用\n\n")
        f.write("评估重点：\n")
        f.write("- 括号/引号是否匹配\n")
        f.write("- 缩进是否正确\n")
        f.write("- 语法结构是否完整\n")
        f.write("- 是否包含必要的配置项\n\n")
        f.write("---\n\n")
        
        for idx, sample in enumerate(samples, 1):
            f.write(f"## 样本 {idx}\n\n")
            f.write(f"**Benchmark**: {sample['benchmark']}\n\n")
            f.write(f"**Name**: {sample['name']}\n\n")
            f.write(f"**自动指标**:\n")
            f.write(f"- Speedup: {sample['speedup']:.3f}x\n")
            f.write(f"- Composite SQ: {sample['composite_sq']:.4f}\n")
            f.write(f"- Off-Structure: {sample['off_structure']:.4f}\n")
            f.write(f"- Below-AR: {'是' if sample['below_ar'] else '否'}\n\n")
            
            f.write("### Prompt (前 200 字符)\n\n")
            f.write("```\n")
            f.write(sample['prompt'][:200])
            if len(sample['prompt']) > 200:
                f.write("\n... (已截断)")
            f.write("\n```\n\n")
            
            f.write("### Reference (前 300 字符)\n\n")
            f.write("```\n")
            f.write(sample['reference'][:300])
            if len(sample['reference']) > 300:
                f.write("\n... (已截断)")
            f.write("\n```\n\n")
            
            f.write("### Generated (前 300 字符)\n\n")
            f.write("```\n")
            f.write(sample['generated'][:300])
            if len(sample['generated']) > 300:
                f.write("\n... (已截断)")
            f.write("\n```\n\n")
            
            f.write("### 评分\n\n")
            f.write("**人工评分** (0/1/2): _____\n\n")
            f.write("**备注**: \n\n")
            f.write("---\n\n")
    
    print(f"✓ 评估表单已生成")

def main():
    """主函数"""
    # 抽取样本
    samples = sample_tasd_fg_results(n_samples=60)
    
    if not samples:
        print("❌ 样本抽取失败")
        return
    
    # 保存样本数据
    results_dir = Path("/root/autodl-tmp/results")
    samples_file = results_dir / "human_quality_review_samples.json"
    
    with open(samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 样本数据已保存: {samples_file}")
    
    # 生成评估表单
    review_form = results_dir / "human_quality_review_form.md"
    generate_review_form(samples, review_form)
    
    print("\n" + "=" * 70)
    print("✅ 人工质量评估准备完成")
    print("=" * 70)
    print("\n下一步:")
    print(f"1. 打开 {review_form.name} 进行人工评估")
    print(f"2. 填写每个样本的评分 (0/1/2)")
    print(f"3. 完成后运行统计脚本生成报告")

if __name__ == "__main__":
    main()
