#!/usr/bin/env python3
"""
论文级图表生成脚本：生成 TASD-FG 论文所需的所有可视化图表

包括：
1. 主实验结果对比图 (6 methods × 6 benchmarks)
2. 消融实验柱状图
3. Speedup vs Quality 散点图
4. Below-AR 案例分析图
5. 256-token 扩展实验对比图
6. LLaMA 泛化实验对比图
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置配色方案
COLORS = {
    'AR': '#95a5a6',
    'N-gram': '#3498db',
    'Greedy': '#9b59b6',
    'FLY': '#e74c3c',
    'TASD': '#2ecc71',
    'TASD-FG': '#f39c12'
}

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

os.makedirs(FIGURES_DIR, exist_ok=True, exist_ok=True)


def load_data():
    """加载所有实验数据"""
    data = {}
    
    # 主实验数据
    with open(f"{RESULTS_DIR}/final_master_report.json") as f:
        data['main'] = json.load(f)
    
    # 消融实验数据
    with open(f"{RESULTS_DIR}/qwen_ablation_7variant.json") as f:
        data['ablation'] = json.load(f)
    
    # 256-token 扩展实验
    if os.path.exists(f"{RESULTS_DIR}/qwen_256token_extended_3x40.json"):
        with open(f"{RESULTS_DIR}/qwen_256token_extended_3x40.json") as f:
            data['extended_256'] = json.load(f)
    
    # LLaMA 泛化实验
    if os.path.exists(f"{RESULTS_DIR}/llama_6x80_full.json"):
        with open(f"{RESULTS_DIR}/llama_6x80_full.json") as f:
            data['llama'] = json.load(f)
    
    return data


def plot_main_results(data):
    """图1: 主实验结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = ['AR', 'N-gram', 'Greedy', 'FLY', 'TASD', 'TASD-FG']
    benchmarks = ['argparse', 'dict_config', 'openmmlab', 'pipeline', 'complex_nested', 'rich_cli']
    
    # 1a. Speedup 对比 (柱状图)
    ax = axes[0, 0]
    x = np.arange(len(benchmarks))
    width = 0.12
    
    for i, method in enumerate(methods):
        speeds = [data['main']['per_benchmark'][bench][method]['sp_mean'] 
                  for bench in benchmarks]
        ax.bar(x + i*width - 2.5*width, speeds, width, label=method, 
               color=COLORS[method], alpha=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('(a) Speedup Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # 1b. Below-AR 统计 (堆叠柱状图)
    ax = axes[0, 1]
    below_counts = {}
    for method in methods:
        below_counts[method] = [data['main']['per_benchmark'][bench][method]['below'] 
                                for bench in benchmarks]
    
    bottom = np.zeros(len(benchmarks))
    for method in methods:
        ax.bar(x, below_counts[method], width=0.6, label=method, 
               color=COLORS[method], alpha=0.8, bottom=bottom)
        bottom += below_counts[method]
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('(b) Below-AR Cases', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 1c. SQ-R vs SQ-S 散点图
    ax = axes[1, 0]
    for method in methods:
        sq_r = [data['main']['per_benchmark'][bench][method]['sq_r'] 
                for bench in benchmarks]
        sq_s = [data['main']['per_benchmark'][bench][method]['sq_s'] 
                for bench in benchmarks]
        ax.scatter(sq_r, sq_s, s=100, label=method, color=COLORS[method], 
                   alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('SQ-R (Reference-aware)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SQ-S (Structure Safety)', fontsize=12, fontweight='bold')
    ax.set_title('(c) Quality Trade-off', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 1d. Overall 性能对比 (雷达图)
    ax = axes[1, 1]
    ax = fig.add_subplot(224, polar=True)
    
    categories = ['Speedup', '1-Below', 'SQ-R', 'SQ-S', '1-OffStr']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 计算总体指标
    overall_metrics = {}
    for method in methods:
        speedup = data['main']['overall'][method]['sp_mean']
        below = data['main']['overall'][method]['below'] / 480
        sq_r = data['main']['overall'][method]['sq_r']
        sq_s = data['main']['overall'][method]['sq_s']
        off_str = data['main']['overall'][method]['off_str']
        
        overall_metrics[method] = [
            speedup / 2.5,  # 归一化到 0-1
            1 - below,
            sq_r,
            sq_s,
            1 - off_str
        ]
    
    for method in methods:
        values = overall_metrics[method] + overall_metrics[method][:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=method, 
                color=COLORS[method], alpha=0.7)
        ax.fill(angles, values, alpha=0.1, color=COLORS[method])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('(d) Overall Performance', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig1_main_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig1_main_results.pdf", bbox_inches='tight')
    print("✓ 图1: 主实验结果对比图已保存")


def plot_ablation_study(data):
    """图2: 消融实验柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = data['ablation']['variants']
    names = [v['name'] for v in variants]
    speedups = [v['sp_mean'] for v in variants]
    belows = [v['below'] for v in variants]
    
    # 2a. Speedup 对比
    ax = axes[0]
    colors = ['#f39c12' if 'TASD-FG' in name else '#2ecc71' if 'TASD' in name 
              else '#95a5a6' for name in names]
    
    bars = ax.barh(names, speedups, color=colors, alpha=0.8, edgecolors='black', linewidth=1)
    ax.set_xlabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('(a) Ablation: Speedup', fontsize=14, fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, speedups)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}×', va='center', fontsize=9, fontweight='bold')
    
    # 2b. Below-AR 对比
    ax = axes[1]
    bars = ax.barh(names, belows, color=colors, alpha=0.8, edgecolors='black', linewidth=1)
    ax.set_xlabel('Below-AR Count', fontsize=12, fontweight='bold')
    ax.set_title('(b) Ablation: Below-AR Cases', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, belows)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                str(val), va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_ablation_study.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig2_ablation_study.pdf", bbox_inches='tight')
    print("✓ 图2: 消融实验柱状图已保存")


def plot_speedup_quality_tradeoff(data):
    """图3: Speedup vs Quality 散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    methods = ['AR', 'N-gram', 'Greedy', 'FLY', 'TASD', 'TASD-FG']
    benchmarks = ['argparse', 'dict_config', 'openmmlab', 'pipeline', 'complex_nested', 'rich_cli']
    
    # 收集所有数据点
    all_points = []
    for method in methods:
        for bench in benchmarks:
            sp = data['main']['per_benchmark'][bench][method]['sp_mean']
            sq_r = data['main']['per_benchmark'][bench][method]['sq_r']
            sq_s = data['main']['per_benchmark'][bench][method]['sq_s']
            composite_sq = (sq_r + sq_s) / 2
            all_points.append({
                'method': method,
                'benchmark': bench,
                'speedup': sp,
                'composite_sq': composite_sq
            })
    
    # 绘制散点图
    for method in methods:
        points = [p for p in all_points if p['method'] == method]
        x = [p['speedup'] for p in points]
        y = [p['composite_sq'] for p in points]
        
        ax.scatter(x, y, s=150, label=method, color=COLORS[method], 
                   alpha=0.7, edgecolors='black', linewidth=1.5, zorder=3)
    
    # 添加 Pareto 前沿
    pareto_points = []
    for method in methods:
        points = [p for p in all_points if p['method'] == method]
        avg_sp = np.mean([p['speedup'] for p in points])
        avg_sq = np.mean([p['composite_sq'] for p in points])
        pareto_points.append((avg_sp, avg_sq, method))
    
    pareto_points.sort(key=lambda x: x[0])
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    
    # 计算 Pareto 前沿
    pareto_front_x = []
    pareto_front_y = []
    max_y = -np.inf
    for x, y in zip(pareto_x, pareto_y):
        if y > max_y:
            pareto_front_x.append(x)
            pareto_front_y.append(y)
            max_y = y
    
    ax.plot(pareto_front_x, pareto_front_y, 'r--', linewidth=2, alpha=0.5, label='Pareto Front')
    
    # 标注平均值
    for method in methods:
        points = [p for p in all_points if p['method'] == method]
        avg_sp = np.mean([p['speedup'] for p in points])
        avg_sq = np.mean([p['composite_sq'] for p in points])
        ax.annotate(method, (avg_sp, avg_sq), textcoords="offset points", 
                    xytext=(10, 10), fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[method], alpha=0.3))
    
    ax.set_xlabel('Speedup (×)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Composite SQ', fontsize=14, fontweight='bold')
    ax.set_title('Speedup vs Quality Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    ax.axvline(x=1.0, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_speedup_quality_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig3_speedup_quality_tradeoff.pdf", bbox_inches='tight')
    print("✓ 图3: Speedup vs Quality 散点图已保存")


def plot_below_ar_analysis(data):
    """图4: Below-AR 案例分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 加载 TASD-FG 详细数据
    with open(f"{RESULTS_DIR}/qwen_tasd_fg_6x80.json") as f:
        tasdfg_data = json.load(f)
    
    # 4a. Below-AR 样本分布
    ax = axes[0, 0]
    below_samples = []
    for bench, bench_data in tasdfg_data['per_benchmark'].items():
        for sample in bench_data['samples']:
            if sample['sp'] < 1.0:
                below_samples.append({
                    'benchmark': bench,
                    'speedup': sample['sp'],
                    'sq_r': sample.get('sq_r', 0),
                    'sq_s': sample.get('sq_s', 0)
                })
    
    if below_samples:
        benchmarks = list(set(s['benchmark'] for s in below_samples))
        counts = [sum(1 for s in below_samples if s['benchmark'] == b) for b in benchmarks]
        
        ax.bar(range(len(benchmarks)), counts, color='#e74c3c', alpha=0.7, edgecolors='black')
        ax.set_xticks(range(len(benchmarks)))
        ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('(a) Below-AR Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # 4b. Speedup 分布直方图
    ax = axes[0, 1]
    all_speedups = []
    for bench, bench_data in tasdfg_data['per_benchmark'].items():
        for sample in bench_data['samples']:
            all_speedups.append(sample['sp'])
    
    ax.hist(all_speedups, bins=30, color='#f39c12', alpha=0.7, edgecolors='black')
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='AR baseline')
    ax.set_xlabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(b) Speedup Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 4c. SQ-R vs SQ-S (Below-AR 样本)
    ax = axes[1, 0]
    if below_samples:
        sq_r = [s['sq_r'] for s in below_samples]
        sq_s = [s['sq_s'] for s in below_samples]
        ax.scatter(sq_r, sq_s, s=100, color='#e74c3c', alpha=0.7, 
                   edgecolors='black', linewidth=1.5, label='Below-AR')
        
        # 对比正常样本
        normal_sq_r = [s.get('sq_r', 0) for bench, bench_data in tasdfg_data['per_benchmark'].items() 
                       for s in bench_data['samples'] if s['sp'] >= 1.0]
        normal_sq_s = [s.get('sq_s', 0) for bench, bench_data in tasdfg_data['per_benchmark'].items() 
                       for s in bench_data['samples'] if s['sp'] >= 1.0]
        ax.scatter(normal_sq_r, normal_sq_s, s=50, color='#2ecc71', alpha=0.3, 
                   label='Normal', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('SQ-R', fontsize=12, fontweight='bold')
        ax.set_ylabel('SQ-S', fontsize=12, fontweight='bold')
        ax.set_title('(c) Below-AR Quality', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    # 4d. Guard 触发统计
    ax = axes[1, 1]
    guard_counts = []
    for bench, bench_data in tasdfg_data['per_benchmark'].items():
        count = sum(1 for s in bench_data['samples'] if s.get('guard_triggered', False))
        guard_counts.append(count)
    
    benchmarks = list(tasdfg_data['per_benchmark'].keys())
    ax.bar(range(len(benchmarks)), guard_counts, color='#3498db', alpha=0.7, edgecolors='black')
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.set_ylabel('Guard Trigger Count', fontsize=12, fontweight='bold')
    ax.set_title('(d) Guard Triggers', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_below_ar_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig4_below_ar_analysis.pdf", bbox_inches='tight')
    print("✓ 图4: Below-AR 案例分析图已保存")


def plot_extended_256_comparison(data):
    """图5: 256-token 扩展实验对比图"""
    if 'extended_256' not in data:
        print("⚠ 跳过图5: 未找到 256-token 扩展实验数据")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['AR', 'FLY', 'TASD', 'TASD-FG']
    benchmarks = list(data['extended_256']['per_benchmark'].keys())
    
    # 5a. Speedup 对比
    ax = axes[0]
    x = np.arange(len(benchmarks))
    width = 0.2
    
    for i, method in enumerate(methods):
        speeds = [data['extended_256']['per_benchmark'][bench][method]['sp_mean'] 
                  for bench in benchmarks]
        ax.bar(x + i*width - 1.5*width, speeds, width, label=method, 
               color=COLORS[method], alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('(a) 256-token Speedup', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # 5b. Quality 对比
    ax = axes[1]
    for i, method in enumerate(methods):
        sq_r = [data['extended_256']['per_benchmark'][bench][method]['sq_r'] 
                for bench in benchmarks]
        sq_s = [data['extended_256']['per_benchmark'][bench][method]['sq_s'] 
                for bench in benchmarks]
        composite_sq = [(r + s) / 2 for r, s in zip(sq_r, sq_s)]
        
        ax.bar(x + i*width - 1.5*width, composite_sq, width, label=method, 
               color=COLORS[method], alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Composite SQ', fontsize=12, fontweight='bold')
    ax.set_title('(b) 256-token Quality', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_extended_256_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig5_extended_256_comparison.pdf", bbox_inches='tight')
    print("✓ 图5: 256-token 扩展实验对比图已保存")


def plot_llama_generalization(data):
    """图6: LLaMA 泛化实验对比图"""
    if 'llama' not in data:
        print("⚠ 跳过图6: 未找到 LLaMA 泛化实验数据")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = ['AR', 'FLY', 'TASD', 'TASD-FG']
    benchmarks = list(data['llama']['per_benchmark'].keys())
    
    # 6a. Speedup 对比
    ax = axes[0]
    x = np.arange(len(benchmarks))
    width = 0.2
    
    for i, method in enumerate(methods):
        speeds = [data['llama']['per_benchmark'][bench][method]['sp_mean'] 
                  for bench in benchmarks]
        ax.bar(x + i*width - 1.5*width, speeds, width, label=method, 
               color=COLORS[method], alpha=0.8, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
    ax.set_title('(a) LLaMA-8B Speedup', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace('_', '\n') for b in benchmarks], fontsize=9)
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # 6b. Qwen vs LLaMA 对比
    ax = axes[1]
    if 'main' in data:
        qwen_speedups = [data['main']['overall']['TASD']['sp_mean'], 
                         data['main']['overall']['TASD-FG']['sp_mean']]
        llama_speedups = [data['llama']['overall']['TASD']['sp_mean'], 
                          data['llama']['overall']['TASD-FG']['sp_mean']]
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, qwen_speedups, width, label='Qwen-14B', 
               color='#3498db', alpha=0.8, edgecolors='black', linewidth=1)
        ax.bar(x + width/2, llama_speedups, width, label='LLaMA-8B', 
               color='#e74c3c', alpha=0.8, edgecolors='black', linewidth=1)
        
        ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax.set_title('(b) Model Generalization', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['TASD', 'TASD-FG'], fontsize=11)
        ax.legend(fontsize=10)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig6_llama_generalization.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{FIGURES_DIR}/fig6_llama_generalization.pdf", bbox_inches='tight')
    print("✓ 图6: LLaMA 泛化实验对比图已保存")


def generate_summary_table(data):
    """生成论文主表 (LaTeX 格式)"""
    
    methods = ['AR', 'N-gram', 'Greedy', 'FLY', 'TASD', 'TASD-FG']
    benchmarks = ['argparse', 'dict_config', 'openmmlab', 'pipeline', 'complex_nested', 'rich_cli']
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Main Results: Speedup and Quality Metrics}")
    latex.append("\\label{tab:main_results}")
    latex.append("\\begin{tabular}{l" + "c" * (len(benchmarks) + 1) + "}")
    latex.append("\\toprule")
    
    # 表头
    header = "Method & " + " & ".join([b.replace('_', ' ').title() for b in benchmarks]) + " & Avg \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # 数据行
    for method in methods:
        row = [method]
        speeds = []
        for bench in benchmarks:
            sp = data['main']['per_benchmark'][bench][method]['sp_mean']
            speeds.append(sp)
            row.append(f"{sp:.3f}")
        avg = np.mean(speeds)
        row.append(f"{avg:.3f}")
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(f"{FIGURES_DIR}/table_main_results.tex", "w") as f:
        f.write("\n".join(latex))
    
    print("✓ 论文主表 (LaTeX) 已保存")


def main():
    print("=" * 70)
    print("论文级图表生成脚本")
    print("=" * 70)
    
    # 加载数据
    print("\n1. 加载实验数据...")
    data = load_data()
    print(f"   已加载: {', '.join(data.keys())}")
    
    # 生成图表
    print("\n2. 生成图表...")
    plot_main_results(data)
    plot_ablation_study(data)
    plot_speedup_quality_tradeoff(data)
    plot_below_ar_analysis(data)
    plot_extended_256_comparison(data)
    plot_llama_generalization(data)
    
    # 生成表格
    print("\n3. 生成论文表格...")
    generate_summary_table(data)
    
    print("\n" + "=" * 70)
    print("所有图表已生成完毕！")
    print(f"保存位置: {FIGURES_DIR}/")
    print("=" * 70)
    
    # 列出生成的文件
    print("\n生成的文件:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.startswith('fig') or f.startswith('table'):
            size = os.path.getsize(f"{FIGURES_DIR}/{f}") / 1024
            print(f"  - {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
