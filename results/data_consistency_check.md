# 数据口径一致性检查报告

生成时间: 2024

## 1. 最终报告数据源

### final_master_report.json

## 2. 256-token 扩展实验

## 3. LLaMA 泛化实验

## 4. 历史报告（旧口径）

以下报告使用旧版 SQ 指标（未拆分为 SQ-R/SQ-S），仅供内部参考：

- ⚠️  `qwen_5method_6x80_quality.json` (旧口径)
- ⚠️  `qwen_tasd_fg_6x80.json` (旧口径)
- ⚠️  `qwen_ablation_7variant.json` (旧口径)

**建议**: 写论文时以 `final_master_report.md` 为准，旧报告标注为阶段性数据。

## 5. 数据一致性结论

✅ **最终报告数据源完整**
✅ **256-token 扩展实验数据完整**
✅ **LLaMA 泛化实验数据完整**
⚠️  **存在旧口径报告，需标注**

### 论文写作建议

1. **主实验数据**: 使用 `final_master_report.md` 中的 Table 1 和 Table 2
2. **256-token 数据**: 使用 `qwen_256token_extended_3x40.md`
3. **LLaMA 泛化数据**: 使用 `llama_6x40_full.md`
4. **消融实验数据**: 使用 `qwen_ablation_7variant.md`
5. **旧报告处理**: 在论文中不引用，仅在附录或补充材料中标注
