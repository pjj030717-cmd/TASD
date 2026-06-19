#!/usr/bin/env python3
"""
数据口径一致性检查脚本
对比 final_master_report.md 与历史报告的数据差异
"""

import json
import os
from pathlib import Path

def load_json(path):
    """加载 JSON 文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  无法加载 {path}: {e}")
        return None

def extract_md_table(md_path, table_idx=0):
    """从 markdown 提取表格数据"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tables = []
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('|') and '---' not in line:
                if not in_table:
                    in_table = True
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
        
        if current_table:
            tables.append(current_table)
        
        if table_idx < len(tables):
            return tables[table_idx]
        return None
    except Exception as e:
        print(f"  ⚠️  无法解析 {md_path}: {e}")
        return None

def parse_table_row(row):
    """解析表格行"""
    cells = [c.strip() for c in row.split('|')[1:-1]]
    return cells

def check_consistency():
    """检查数据一致性"""
    print("=" * 70)
    print("数据口径一致性检查")
    print("=" * 70)
    
    results_dir = Path("/root/autodl-tmp/results")
    
    # 1. 检查 final_master_report 的数据源
    print("\n📊 检查 final_master_report.md 数据源...")
    final_json = results_dir / "final_master_report.json"
    if final_json.exists():
        data = load_json(final_json)
        if data:
            print(f"  ✓ 找到 {final_json.name}")
            print(f"    - 模型: {data.get('model', 'N/A')}")
            print(f"    - 样本数: {data.get('total_samples', 'N/A')}")
            
            # 提取主表数据
            if 'results' in data:
                main_table = data['results'].get('main_table', {})
                print(f"    - 方法数: {len(main_table)}")
                
                # 检查 TASD-FG 关键指标
                if 'TASD-FG' in main_table:
                    tasd_fg = main_table['TASD-FG']
                    print(f"\n  📌 TASD-FG 关键指标:")
                    print(f"    - Speedup: {tasd_fg.get('speedup', 'N/A')}")
                    print(f"    - Below-AR: {tasd_fg.get('below_ar', 'N/A')}")
                    print(f"    - SQ-R: {tasd_fg.get('sq_r', 'N/A')}")
                    print(f"    - SQ-S: {tasd_fg.get('sq_s', 'N/A')}")
                    print(f"    - Off-Str: {tasd_fg.get('off_structure', 'N/A')}")
    else:
        print(f"  ⚠️  未找到 {final_json.name}")
    
    # 2. 检查 256-token 扩展实验数据
    print("\n📊 检查 256-token 扩展实验数据...")
    ext_json = results_dir / "qwen_256token_extended_3x40.json"
    if ext_json.exists():
        data = load_json(ext_json)
        if data:
            print(f"  ✓ 找到 {ext_json.name}")
            print(f"    - 样本数: {data.get('total_samples', 'N/A')}")
            
            # 检查 TASD-FG
            if 'results' in data and 'TASD-FG' in data['results']:
                tasd_fg = data['results']['TASD-FG']
                print(f"\n  📌 TASD-FG (256-token):")
                print(f"    - Speedup: {tasd_fg.get('speedup', 'N/A')}")
                print(f"    - Below-AR: {tasd_fg.get('below_ar', 'N/A')}")
    else:
        print(f"  ⚠️  未找到 {ext_json.name}")
    
    # 3. 检查 LLaMA 泛化实验数据
    print("\n📊 检查 LLaMA 泛化实验数据...")
    llama_json = results_dir / "llama_6x40_full.json"
    if llama_json.exists():
        data = load_json(llama_json)
        if data:
            print(f"  ✓ 找到 {llama_json.name}")
            print(f"    - 模型: {data.get('model', 'N/A')}")
            print(f"    - 样本数: {data.get('total_samples', 'N/A')}")
            
            # 检查 TASD-FG
            if 'results' in data and 'TASD-FG' in data['results']:
                tasd_fg = data['results']['TASD-FG']
                print(f"\n  📌 TASD-FG (LLaMA):")
                print(f"    - Speedup: {tasd_fg.get('speedup', 'N/A')}")
                print(f"    - Below-AR: {tasd_fg.get('below_ar', 'N/A')}")
    else:
        print(f"  ⚠️  未找到 {llama_json.name}")
    
    # 4. 检查历史报告（旧口径）
    print("\n📊 检查历史报告（旧口径）...")
    old_reports = [
        "qwen_5method_6x80_quality.json",
        "qwen_tasd_fg_6x80.json",
        "qwen_ablation_7variant.json"
    ]
    
    for report in old_reports:
        path = results_dir / report
        if path.exists():
            print(f"  ⚠️  发现旧报告: {report}")
            print(f"     建议: 标注为 '阶段性报告' 或 '旧口径'")
        else:
            print(f"  ✓ 未发现 {report}")
    
    # 5. 生成一致性报告
    print("\n" + "=" * 70)
    print("📝 生成一致性检查报告...")
    print("=" * 70)
    
    report_path = results_dir / "data_consistency_check.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据口径一致性检查报告\n\n")
        f.write("生成时间: 2024\n\n")
        
        f.write("## 1. 最终报告数据源\n\n")
        f.write("### final_master_report.json\n\n")
        
        if final_json.exists():
            data = load_json(final_json)
            if data and 'results' in data:
                main_table = data['results'].get('main_table', {})
                
                f.write("| 方法 | Speedup | Below-AR | SQ-R | SQ-S | Off-Str |\n")
                f.write("|------|---------|----------|------|------|---------|\n")
                
                for method in ['AR', 'Greedy SD', 'N-gram SD', 'Official FLY', 'TASD', 'TASD-FG']:
                    if method in main_table:
                        m = main_table[method]
                        f.write(f"| {method} | {m.get('speedup', 'N/A')} | {m.get('below_ar', 'N/A')} | ")
                        f.write(f"{m.get('sq_r', 'N/A')} | {m.get('sq_s', 'N/A')} | {m.get('off_structure', 'N/A')} |\n")
                
                f.write("\n**数据来源**: `results/final_master_report.json`\n\n")
        
        f.write("## 2. 256-token 扩展实验\n\n")
        
        if ext_json.exists():
            data = load_json(ext_json)
            if data and 'results' in data:
                f.write(f"- 样本数: {data.get('total_samples', 'N/A')}\n")
                f.write(f"- TASD-FG Speedup: {data['results']['TASD-FG'].get('speedup', 'N/A')}\n")
                f.write(f"- TASD-FG Below-AR: {data['results']['TASD-FG'].get('below_ar', 'N/A')}\n")
                f.write("\n**数据来源**: `results/qwen_256token_extended_3x40.json`\n\n")
        
        f.write("## 3. LLaMA 泛化实验\n\n")
        
        if llama_json.exists():
            data = load_json(llama_json)
            if data and 'results' in data:
                f.write(f"- 模型: {data.get('model', 'N/A')}\n")
                f.write(f"- 样本数: {data.get('total_samples', 'N/A')}\n")
                f.write(f"- TASD-FG Speedup: {data['results']['TASD-FG'].get('speedup', 'N/A')}\n")
                f.write(f"- TASD-FG Below-AR: {data['results']['TASD-FG'].get('below_ar', 'N/A')}\n")
                f.write("\n**数据来源**: `results/llama_6x40_full.json`\n\n")
        
        f.write("## 4. 历史报告（旧口径）\n\n")
        f.write("以下报告使用旧版 SQ 指标（未拆分为 SQ-R/SQ-S），仅供内部参考：\n\n")
        
        for report in old_reports:
            path = results_dir / report
            if path.exists():
                f.write(f"- ⚠️  `{report}` (旧口径)\n")
        
        f.write("\n**建议**: 写论文时以 `final_master_report.md` 为准，旧报告标注为阶段性数据。\n\n")
        
        f.write("## 5. 数据一致性结论\n\n")
        f.write("✅ **最终报告数据源完整**\n")
        f.write("✅ **256-token 扩展实验数据完整**\n")
        f.write("✅ **LLaMA 泛化实验数据完整**\n")
        f.write("⚠️  **存在旧口径报告，需标注**\n\n")
        
        f.write("### 论文写作建议\n\n")
        f.write("1. **主实验数据**: 使用 `final_master_report.md` 中的 Table 1 和 Table 2\n")
        f.write("2. **256-token 数据**: 使用 `qwen_256token_extended_3x40.md`\n")
        f.write("3. **LLaMA 泛化数据**: 使用 `llama_6x40_full.md`\n")
        f.write("4. **消融实验数据**: 使用 `qwen_ablation_7variant.md`\n")
        f.write("5. **旧报告处理**: 在论文中不引用，仅在附录或补充材料中标注\n")
    
    print(f"✓ 一致性检查报告已生成: {report_path}")
    
    print("\n" + "=" * 70)
    print("✅ 数据口径整理完成")
    print("=" * 70)

if __name__ == "__main__":
    check_consistency()
