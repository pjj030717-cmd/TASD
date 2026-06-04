"""
Final Structure Coverage Scan.

Scans real code files for structured code-completion opportunities.
Does NOT run any models — static analysis only.

Covers 9 structure types:
  Validated: argparse, rich_cli_option_groups, dict_config,
             complex_nested_config, openmmlab_config, pipeline_stage_config
  Boundary:  schema_fields, model_fields, pytest_parametrize
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = "/root/autodl-tmp/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SOURCE_DIRS = [
    "/root/miniconda3/lib/python3.12/site-packages",
    "/root/autodl-tmp/benchmark_sources/openmmlab",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmsegmentation",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmpretrain",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmengine",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmpose",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmocr",
]


# ============================================================
# Utility
# ============================================================

def _find_python_files(dirs, limit=None):
    py_files = []
    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            if "__pycache__" in root or "/tests/" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
                    if limit and len(py_files) >= limit:
                        return py_files
    return py_files


def _get_relative_path(filepath):
    for src in SOURCE_DIRS:
        if filepath.startswith(src):
            return filepath[len(src):].lstrip("/")
    return filepath


def _get_source_repo(filepath):
    if "openmmlab_mmsegmentation" in filepath:
        return "open-mmlab/mmsegmentation"
    elif "openmmlab_mmpretrain" in filepath:
        return "open-mmlab/mmpretrain"
    elif "openmmlab_mmengine" in filepath:
        return "open-mmlab/mmengine"
    elif "openmmlab_mmpose" in filepath:
        return "open-mmlab/mmpose"
    elif "openmmlab_mmocr" in filepath:
        return "open-mmlab/mmocr"
    elif "openmmlab" in filepath:
        return "open-mmlab/mmdetection"
    else:
        rel = _get_relative_path(filepath)
        parts = rel.split("/")
        if parts:
            return f"python-{parts[0]}"
        return "python-unknown"


def _count_lines(text):
    return len(text.strip().split("\n")) if text.strip() else 0


def _count_chars(text):
    return len(text)


# ============================================================
# Structure Type Definitions & Extractors
# ============================================================

STRUCTURE_DEFS = {}


# ---- argparse ----
def _extract_argparse(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    i = 0
    pattern = r'\.(add_argument|add_option)\s*\('
    while i < len(lines):
        line = lines[i]
        if re.search(pattern, line):
            start = i
            all_opt_lines = []
            j = i
            while j < len(lines) and j < i + 50:
                if re.search(pattern, lines[j]):
                    call_lines = [lines[j]]
                    depth = lines[j].count("(") - lines[j].count(")")
                    k = j + 1
                    while k < len(lines) and depth > 0:
                        call_lines.append(lines[k])
                        depth += lines[k].count("(") - lines[k].count(")")
                        k += 1
                    all_opt_lines.extend(call_lines)
                    j = k
                elif lines[j].strip() and not lines[j].strip().startswith("#"):
                    break
                else:
                    j += 1
            arg_count = len(re.findall(pattern, "".join(all_opt_lines)))
            if arg_count >= 3:
                # Split: first ~40-60% as prompt
                total = len(all_opt_lines)
                split_pt = max(2, min(int(total * 0.5), total - 3))
                prompt = "".join(all_opt_lines[:split_pt]).rstrip()
                reference = "".join(all_opt_lines[split_pt:]).rstrip()
                seed_c = len(re.findall(pattern, prompt))
                ref_c = len(re.findall(pattern, reference))
                blocks.append({
                    "lines": all_opt_lines,
                    "start_line": start,
                    "prompt": prompt,
                    "reference": reference,
                    "seed_count": seed_c,
                    "ref_structure_count": ref_c,
                    "nesting_depth": 0,
                })
            i = j
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["argparse"] = {
    "display": "Argparse Option Blocks",
    "category": "validated",
    "extractor": _extract_argparse,
    "min_block_lines": 3,
    "min_prompt_lines": 2,
    "min_ref_lines": 2,
    "min_seed": 2,
    "min_ref_structure": 1,
}


# ---- rich_cli_option_groups ----
def _extract_rich_cli(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    patterns = [r'\.(add_argument|add_option)\s*\(', r'click\.option\s*\(', r'typer\.Option\s*\(']
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat in patterns:
            if re.search(pat, line):
                start = i
                all_lines = []
                j = i
                while j < len(lines) and j < i + 100:
                    if any(re.search(p, lines[j]) for p in patterns):
                        call_lines = [lines[j]]
                        depth = lines[j].count("(") - lines[j].count(")")
                        depth += lines[j].count("[") - lines[j].count("]")
                        k = j + 1
                        while k < len(lines) and depth > 0:
                            call_lines.append(lines[k])
                            depth += lines[k].count("(") - lines[k].count(")")
                            depth += lines[k].count("[") - lines[k].count("]")
                            k += 1
                        all_lines.extend(call_lines)
                        j = k
                    elif lines[j].strip() and not lines[j].strip().startswith("#"):
                        break
                    else:
                        j += 1
                opt_count = len(re.findall(pat, "".join(all_lines)))
                has_rich = bool(re.search(r'(choices|default|type|action|help|required|nargs|metavar)\s*=', "".join(all_lines)))
                if opt_count >= 6 and has_rich:
                    # Split by option boundaries
                    opt_positions = [idx for idx, ln in enumerate(all_lines) if any(re.search(p, ln) for p in patterns)]
                    split_idx = opt_positions[min(2, len(opt_positions) - 2)] if len(opt_positions) >= 3 else len(all_lines) // 2
                    prompt = "".join(all_lines[:split_idx]).rstrip()
                    reference = "".join(all_lines[split_idx:]).rstrip()
                    seed_c = len(re.findall(pat, prompt))
                    ref_c = len(re.findall(pat, reference))
                    blocks.append({
                        "lines": all_lines,
                        "start_line": start,
                        "prompt": prompt,
                        "reference": reference,
                        "seed_count": seed_c,
                        "ref_structure_count": ref_c,
                        "nesting_depth": 0,
                    })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["rich_cli_option_groups"] = {
    "display": "Rich CLI Option Groups",
    "category": "validated",
    "extractor": _extract_rich_cli,
    "min_block_lines": 6,
    "min_prompt_lines": 3,
    "min_ref_lines": 3,
    "min_seed": 2,
    "min_ref_structure": 1,
}


# ---- dict_config ----
def _extract_dict_config(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    patterns = [(r'^(\w+)\s*=\s*\{', '{'), (r'^(\w+)\s*=\s*dict\s*\(', 'dict('), (r'^(\w+)\s*=\s*\[', '[')]
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat, btype in patterns:
            match = re.match(pat, line.strip())
            if match:
                vname = match.group(1)
                if vname in ('result', 'ret', 'tmp', 'x', 'y', 'data'):
                    continue
                start = i
                blines = [line]
                if btype == '{':
                    depth = line.count('{') - line.count('}')
                elif btype == 'dict(':
                    depth = line.count('(') - line.count(')')
                else:
                    depth = line.count('[') - line.count(']')
                j = i + 1
                while j < len(lines) and depth > 0:
                    blines.append(lines[j])
                    if btype == '{':
                        depth += lines[j].count('{') - lines[j].count('}')
                    elif btype == 'dict(':
                        depth += lines[j].count('(') - lines[j].count(')')
                    else:
                        depth += lines[j].count('[') - lines[j].count(']')
                    j += 1
                if len(blines) >= 8:
                    btext = "\n".join(blines)
                    if re.search(r'["\']\w+["\']\s*[:=]', btext):
                        prompt_len = min(6, max(3, len(blines) // 3))
                        prompt = "\n".join(blines[:prompt_len])
                        reference = "\n".join(blines[prompt_len:])
                        # nesting depth
                        max_d = 0; cur = 0
                        for ch in btext:
                            if ch in '{(':
                                cur += 1; max_d = max(max_d, cur)
                            elif ch in '})':
                                cur -= 1
                        blocks.append({
                            "lines": blines,
                            "start_line": start,
                            "prompt": prompt,
                            "reference": reference,
                            "seed_count": prompt_len,
                            "ref_structure_count": len(blines) - prompt_len,
                            "nesting_depth": max_d,
                        })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["dict_config"] = {
    "display": "Dict Config Blocks",
    "category": "validated",
    "extractor": _extract_dict_config,
    "min_block_lines": 8,
    "min_prompt_lines": 3,
    "min_ref_lines": 5,
    "min_seed": 1,
    "min_ref_structure": 1,
}


# ---- complex_nested_config ----
def _extract_complex_nested(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    patterns = [
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG|options|OPTIONS|args|ARGS)\s*=\s*\{', '{'),
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG|options|OPTIONS|args|ARGS)\s*=\s*dict\s*\(', 'dict('),
        (r'^\s*dict\s*\(\s*\w+\s*=\s*dict\s*\(', 'nd'),
        (r'^\s*\{\s*["\']\w+["\']\s*:\s*\{', 'nl'),
        (r'^\s*["\']\w+["\']\s*:\s*\{', 'in'),
        (r'^\s*\w+\s*=\s*\{', 'ad'),
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat, btype in patterns:
            if re.search(pat, line.strip()):
                start = i
                blines = [line]
                if btype == '{':
                    depth = line.count('{') - line.count('}')
                elif btype == 'dict(':
                    depth = line.count('(') - line.count(')')
                else:
                    depth = line.count('{') - line.count('}')
                    depth += line.count('(') - line.count(')')
                j = i + 1
                while j < len(lines) and depth > 0:
                    blines.append(lines[j])
                    if btype in ('{', 'nd', 'nl', 'in', 'ad'):
                        depth += lines[j].count('{') - lines[j].count('}')
                    if btype in ('dict(', 'nd'):
                        depth += lines[j].count('(') - lines[j].count(')')
                    j += 1
                if len(blines) >= 8:
                    btext = "".join(blines)
                    max_d = 0; cur = 0
                    for ch in btext:
                        if ch in '{(':
                            cur += 1; max_d = max(max_d, cur)
                        elif ch in '})':
                            cur -= 1
                    if max_d >= 3:  # 2+ nesting levels
                        prompt_len = min(8, max(4, len(blines) // 3))
                        prompt = "\n".join(blines[:prompt_len]).rstrip()
                        reference = "\n".join(blines[prompt_len:]).rstrip()
                        blocks.append({
                            "lines": blines,
                            "start_line": start,
                            "prompt": prompt,
                            "reference": reference,
                            "seed_count": prompt_len,
                            "ref_structure_count": len(blines) - prompt_len,
                            "nesting_depth": max_d,
                        })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["complex_nested_config"] = {
    "display": "Complex Nested Configs",
    "category": "validated",
    "extractor": _extract_complex_nested,
    "min_block_lines": 8,
    "min_prompt_lines": 4,
    "min_ref_lines": 8,
    "min_seed": 1,
    "min_ref_structure": 1,
}


# ---- openmmlab_config ----
def _extract_openmmlab(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('import')]
    if len(code_lines) < 8:
        return []
    blocks = []
    patterns = [
        r'^(model|train_pipeline|test_pipeline|val_pipeline|train_dataloader|val_dataloader|test_dataloader|optim_wrapper|param_scheduler|default_hooks|custom_hooks|log_processor|env|launcher)\s*=\s*',
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat in patterns:
            if re.match(pat, line.strip()):
                start = i
                blines = [line]
                depth = line.count('(') - line.count(')') + line.count('[') - line.count(']') + line.count('{') - line.count('}')
                j = i + 1
                while j < len(lines) and depth > 0:
                    blines.append(lines[j])
                    depth += lines[j].count('(') - lines[j].count(')')
                    depth += lines[j].count('[') - lines[j].count(']')
                    depth += lines[j].count('{') - lines[j].count('}')
                    j += 1
                if len(blines) >= 8:
                    prompt_len = min(6, max(3, len(blines) // 3))
                    prompt = "\n".join(blines[:prompt_len])
                    reference = "\n".join(blines[prompt_len:])
                    blocks.append({
                        "lines": blines,
                        "start_line": start,
                        "prompt": prompt,
                        "reference": reference,
                        "seed_count": prompt_len,
                        "ref_structure_count": len(blines) - prompt_len,
                        "nesting_depth": 0,
                    })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["openmmlab_config"] = {
    "display": "OpenMMLab Configs",
    "category": "validated",
    "extractor": _extract_openmmlab,
    "min_block_lines": 8,
    "min_prompt_lines": 3,
    "min_ref_lines": 5,
    "min_seed": 1,
    "min_ref_structure": 1,
}


# ---- pipeline_stage_config ----
def _extract_pipeline_stage(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    patterns = [
        r'(train_pipeline|test_pipeline|val_pipeline|pipeline|transforms|stages|processors)\s*=\s*\[',
        r'Compose\s*\(\s*\[',
        r'dict\s*\(\s*type\s*=',
        r'\{\s*["\']type["\']\s*:',
        r'\{\s*["\']name["\']\s*:',
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat in patterns:
            if re.search(pat, line):
                start = i
                blines = [line]
                depth = line.count('[') - line.count(']')
                depth += line.count('(') - line.count(')')
                depth += line.count('{') - line.count('}')
                j = i + 1
                while j < len(lines) and depth > 0:
                    blines.append(lines[j])
                    depth += lines[j].count('[') - lines[j].count(']')
                    depth += lines[j].count('(') - lines[j].count(')')
                    depth += lines[j].count('{') - lines[j].count('}')
                    j += 1
                if len(blines) >= 8:
                    btext = "".join(blines)
                    if re.search(r'(type|name)\s*[:=]', btext):
                        prompt_len = min(6, max(3, len(blines) // 3))
                        prompt = "".join(blines[:prompt_len]).rstrip()
                        reference = "".join(blines[prompt_len:]).rstrip()
                        seed_c = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', prompt))
                        ref_c = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', reference))
                        blocks.append({
                            "lines": blines,
                            "start_line": start,
                            "prompt": prompt,
                            "reference": reference,
                            "seed_count": seed_c,
                            "ref_structure_count": ref_c,
                            "nesting_depth": 0,
                        })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["pipeline_stage_config"] = {
    "display": "Pipeline & Stage Configs",
    "category": "validated",
    "extractor": _extract_pipeline_stage,
    "min_block_lines": 8,
    "min_prompt_lines": 3,
    "min_ref_lines": 5,
    "min_seed": 1,
    "min_ref_structure": 2,
}


# ---- Boundary structures ----
# schema_fields: SQLAlchemy/DRF model field definitions
def _extract_schema_fields(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    patterns = [
        r'\bColumn\s*\(',
        r'\b(Integer|String|Float|Boolean|DateTime|Text|ForeignKey)\s*\(',
        r'\bmodels\.(CharField|IntegerField|FloatField|BooleanField|DateTimeField|TextField|ForeignKey|ManyToManyField)\s*\(',
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        for pat in patterns:
            if re.search(pat, line):
                start = i
                all_lines = []
                j = i
                while j < len(lines) and j < i + 30:
                    if any(re.search(p, lines[j]) for p in patterns):
                        call_lines = [lines[j]]
                        depth = lines[j].count("(") - lines[j].count(")")
                        k = j + 1
                        while k < len(lines) and depth > 0:
                            call_lines.append(lines[k])
                            depth += lines[k].count("(") - lines[k].count(")")
                            k += 1
                        all_lines.extend(call_lines)
                        j = k
                    elif lines[j].strip() and not lines[j].strip().startswith("#"):
                        break
                    else:
                        j += 1
                field_count = len(re.findall(r'\b(Column|Integer|String|Float|Boolean|DateTime|TextField|CharField|IntegerField)\s*\(', "".join(all_lines)))
                if field_count >= 3:
                    total = len(all_lines)
                    split_pt = max(2, min(int(total * 0.5), total - 3))
                    prompt = "".join(all_lines[:split_pt]).rstrip()
                    reference = "".join(all_lines[split_pt:]).rstrip()
                    seed_c = len(re.findall(r'\b(Column|Integer|String|Float|Boolean|DateTime|TextField|CharField|IntegerField)\s*\(', prompt))
                    ref_c = len(re.findall(r'\b(Column|Integer|String|Float|Boolean|DateTime|TextField|CharField|IntegerField)\s*\(', reference))
                    blocks.append({
                        "lines": all_lines,
                        "start_line": start,
                        "prompt": prompt,
                        "reference": reference,
                        "seed_count": seed_c,
                        "ref_structure_count": ref_c,
                        "nesting_depth": 0,
                    })
                i = j
                break
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["schema_fields"] = {
    "display": "Schema/Model Fields (SQLAlchemy/DRF)",
    "category": "boundary",
    "extractor": _extract_schema_fields,
    "min_block_lines": 3,
    "min_prompt_lines": 2,
    "min_ref_lines": 2,
    "min_seed": 2,
    "min_ref_structure": 1,
}


# model_fields: Pydantic/Dataclass/NamedTuple/TypedDict fields
def _extract_model_fields(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    # Patterns for data model class definitions
    class_pattern = re.compile(
        r'class\s+\w+\s*\((.*BaseModel.*|.*TypedDict.*|.*NamedTuple.*|.*Protocol.*|.*ABC.*)\)|'
        r'^@dataclass|^@define'
    )
    # Field patterns: type annotated members
    field_pattern = re.compile(
        r'^\s+\w+\s*:\s*(int|str|float|bool|Optional|List|Dict|Union|Any|tuple|set|frozenset|bytes|'
        r'Annotated|Literal|Final|ClassVar|Sequence|Mapping|Callable|Tuple)\b'
    )
    i = 0
    while i < len(lines):
        line = lines[i]
        if class_pattern.search(line):
            start = i
            blines = [line]
            j = i + 1
            # Collect class body
            while j < len(lines):
                stripped = lines[j].strip()
                # Stop at next top-level class/func (not indented, not empty, not comment, not decorator)
                if stripped and not stripped.startswith('#') and not stripped.startswith('@') \
                        and not lines[j][0] in (' ', '\t'):
                    break
                blines.append(lines[j])
                j += 1
            field_count = len(field_pattern.findall("".join(blines)))
            if field_count >= 2:
                total = len(blines)
                split_pt = max(2, min(int(total * 0.5), total - 2))
                prompt = "".join(blines[:split_pt]).rstrip()
                reference = "".join(blines[split_pt:]).rstrip()
                seed_c = len(field_pattern.findall(prompt))
                ref_c = len(field_pattern.findall(reference))
                blocks.append({
                    "lines": blines,
                    "start_line": start,
                    "prompt": prompt,
                    "reference": reference,
                    "seed_count": seed_c,
                    "ref_structure_count": ref_c,
                    "nesting_depth": 0,
                })
            i = j
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["model_fields"] = {
    "display": "Model Fields (Pydantic/Dataclass)",
    "category": "boundary",
    "extractor": _extract_model_fields,
    "min_block_lines": 3,
    "min_prompt_lines": 2,
    "min_ref_lines": 2,
    "min_seed": 2,
    "min_ref_structure": 1,
}


# pytest_parametrize
def _extract_pytest_parametrize(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []
    blocks = []
    pattern = r'@pytest\.mark\.parametrize\s*\('
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.search(pattern, line):
            start = i
            blines = [line]
            depth = line.count("(") - line.count(")")
            j = i + 1
            while j < len(lines) and depth > 0:
                blines.append(lines[j])
                depth += lines[j].count("(") - lines[j].count(")")
                j += 1
            if len(blines) >= 2:
                prompt = "".join(blines[:1]).rstrip()
                reference = "".join(blines[1:]).rstrip()
                blocks.append({
                    "lines": blines,
                    "start_line": start,
                    "prompt": prompt,
                    "reference": reference,
                    "seed_count": 1,
                    "ref_structure_count": len(blines) - 1,
                    "nesting_depth": 0,
                })
            i = j
        else:
            i += 1
    return blocks

STRUCTURE_DEFS["pytest_parametrize"] = {
    "display": "Pytest Parametrize Decorators",
    "category": "boundary",
    "extractor": _extract_pytest_parametrize,
    "min_block_lines": 2,
    "min_prompt_lines": 1,
    "min_ref_lines": 1,
    "min_seed": 1,
    "min_ref_structure": 1,
}


# ============================================================
# Validation Helper
# ============================================================

def is_valid_candidate(block, structure_def):
    """Check whether a candidate block passes filter rules."""
    sdef = structure_def
    prompt = block["prompt"]
    reference = block["reference"]
    prompt_lines = _count_lines(prompt)
    ref_lines = _count_lines(reference)

    # Check prompt quality: not just comments/imports
    filtered_prompt = [l for l in prompt.split("\n") if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("import")]
    if len(filtered_prompt) < sdef["min_seed"]:
        return False, "invalid_prompt_seed"

    # Check reference quality
    filtered_ref = [l for l in reference.split("\n") if l.strip() and not l.strip().startswith("#")]
    if len(filtered_ref) < sdef["min_ref_lines"]:
        return False, "reference_too_short"

    # Check structural seeds
    if block.get("seed_count", 0) < sdef["min_seed"]:
        return False, "invalid_prompt_seed"

    # Check reference structure
    if block.get("ref_structure_count", 0) < sdef["min_ref_structure"]:
        return False, "invalid_reference"

    return True, "valid"


# ============================================================
# Main Scan
# ============================================================

def run_coverage_scan():
    print("Final Structure Coverage Scan")
    print("=" * 60)
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"Source files found: {len(py_files)}")

    coverage = {}
    all_results = {}

    for struct_name, sdef in STRUCTURE_DEFS.items():
        print(f"\n--- Scanning: {sdef['display']} ({struct_name}) ---")
        extractor = sdef["extractor"]

        all_blocks = []
        seen = set()
        raw_count = 0

        for filepath in py_files:
            blocks = extractor(filepath)
            for block in blocks:
                raw_count += 1
                key = (filepath, block["start_line"])
                if key in seen:
                    continue
                seen.add(key)
                block["filepath"] = filepath
                all_blocks.append(block)

        print(f"  Raw: {raw_count}, Unique: {len(all_blocks)}")

        valid_blocks = []
        invalid_seed = 0
        invalid_ref = 0
        ref_too_short = 0

        for block in all_blocks:
            ok, reason = is_valid_candidate(block, sdef)
            if ok:
                valid_blocks.append(block)
            elif reason == "invalid_prompt_seed":
                invalid_seed += 1
            elif reason == "invalid_reference":
                invalid_ref += 1
            elif reason == "reference_too_short":
                ref_too_short += 1

        # Collect stats
        files = list(set(b["filepath"] for b in all_blocks))
        repos = list(set(_get_source_repo(b["filepath"]) for b in all_blocks))

        block_lines = [len(b["lines"]) for b in all_blocks]
        prompt_lines = [_count_lines(b["prompt"]) for b in all_blocks]
        ref_lines = [_count_lines(b["reference"]) for b in all_blocks]
        prompt_chars = [_count_chars(b["prompt"]) for b in all_blocks]
        ref_chars = [_count_chars(b["reference"]) for b in all_blocks]
        depths = [b.get("nesting_depth", 0) for b in all_blocks]
        seeds = [b.get("seed_count", 0) for b in all_blocks]
        ref_structs = [b.get("ref_structure_count", 0) for b in all_blocks]

        def avg(lst):
            return round(sum(lst) / max(len(lst), 1), 2)

        suitabilities = {
            "argparse": "High — repeated skeleton, rich fields, Guard detectable",
            "rich_cli_option_groups": "High — complex multi-option groups, Guard benefits",
            "dict_config": "Medium-High — key/value repetition, nested structures",
            "complex_nested_config": "High — deep nesting, TASD Guard has clear advantage",
            "openmmlab_config": "High — repeated config blocks, dense structure",
            "pipeline_stage_config": "High — repeated stage skeletons with type/name/params",
            "schema_fields": "Low — fields are short, single-line, baseline already good",
            "model_fields": "Low — fields are short, single-line, baseline already good",
            "pytest_parametrize": "Low — nested strings/test values vary widely, Guard error-prone",
        }

        result = {
            "structure_type": struct_name,
            "display": sdef["display"],
            "category": sdef["category"],
            "raw_candidate_count": raw_count,
            "valid_candidate_count": len(valid_blocks),
            "unique_candidate_count": len(all_blocks),
            "source_file_count": len(files),
            "repo_or_package_count": len(repos),
            "invalid_prompt_seed_count": invalid_seed,
            "invalid_reference_count": invalid_ref,
            "reference_too_short_count": ref_too_short,
            "duplicate_removed_count": raw_count - len(all_blocks),
            "avg_block_lines": avg(block_lines),
            "avg_prompt_lines": avg(prompt_lines),
            "avg_reference_lines": avg(ref_lines),
            "avg_prompt_chars": avg(prompt_chars),
            "avg_reference_chars": avg(ref_chars),
            "avg_nesting_depth": avg(depths),
            "avg_seed_count": avg(seeds),
            "avg_reference_structure_count": avg(ref_structs),
            "suitability": suitabilities.get(struct_name, "Unknown"),
        }

        coverage[struct_name] = result
        print(f"  Valid: {len(valid_blocks)}, Files: {len(files)}, Repos: {len(repos)}")

    all_results["coverage"] = coverage
    all_results["scan_info"] = {
        "total_source_files": len(py_files),
        "source_dirs": SOURCE_DIRS,
        "structure_types_scanned": list(STRUCTURE_DEFS.keys()),
        "validated_structures": [k for k, v in STRUCTURE_DEFS.items() if v["category"] == "validated"],
        "boundary_structures": [k for k, v in STRUCTURE_DEFS.items() if v["category"] == "boundary"],
    }

    # Save JSON
    json_path = os.path.join(RESULTS_DIR, "final_structure_coverage_scan.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {json_path}")

    return all_results


def generate_md_report(coverage_data):
    """Generate Markdown report from coverage data."""
    cov = coverage_data["coverage"]
    info = coverage_data["scan_info"]

    lines = []
    lines.append("# Final Structure Coverage Scan Report")
    lines.append("")
    lines.append(f"**Source files scanned**: {info['total_source_files']}")
    lines.append(f"**Structure types assessed**: {len(info['structure_types_scanned'])}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Overall Coverage Table")
    lines.append("")
    lines.append("| Structure Type | Raw | Valid | Files | Repos | Avg Block Lines | Avg Seed | Avg Ref Struct | Suitability |")
    lines.append("|----------------|-----|-------|-------|-------|-----------------|----------|----------------|-------------|")

    # Order: validated first, then boundary
    ordered = (
        [k for k in info["validated_structures"] if k in cov] +
        [k for k in info["boundary_structures"] if k in cov]
    )

    for key in ordered:
        c = cov[key]
        suit_short = c["suitability"].split("—")[0].strip()
        lines.append(
            f"| {c['display']} | {c['raw_candidate_count']} | {c['valid_candidate_count']} | "
            f"{c['source_file_count']} | {c['repo_or_package_count']} | "
            f"{c['avg_block_lines']} | {c['avg_seed_count']} | {c['avg_reference_structure_count']} | {suit_short} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Valid Benchmark-Ready Structures")
    lines.append("")
    lines.append("These structures have sufficient valid candidates and are suitable for TASD evaluation.")
    lines.append("")

    for key in info["validated_structures"]:
        if key not in cov:
            continue
        c = cov[key]
        lines.append(f"### {c['display']} (`{key}`)")
        lines.append("")
        lines.append(f"- **Raw candidates**: {c['raw_candidate_count']}")
        lines.append(f"- **Valid candidates**: {c['valid_candidate_count']} (from {c['unique_candidate_count']} unique)")
        lines.append(f"- **Source files**: {c['source_file_count']}")
        lines.append(f"- **Repos/packages**: {c['repo_or_package_count']}")
        lines.append(f"- **Avg block lines**: {c['avg_block_lines']}")
        lines.append(f"- **Avg prompt lines**: {c['avg_prompt_lines']} ({c['avg_prompt_chars']} chars)")
        lines.append(f"- **Avg reference lines**: {c['avg_reference_lines']} ({c['avg_reference_chars']} chars)")
        lines.append(f"- **Avg nesting depth**: {c['avg_nesting_depth']}")
        lines.append(f"- **Avg seed count**: {c['avg_seed_count']}")
        lines.append(f"- **Avg ref structure count**: {c['avg_reference_structure_count']}")
        lines.append(f"- **Filtered — invalid seed**: {c['invalid_prompt_seed_count']}")
        lines.append(f"- **Filtered — invalid reference**: {c['invalid_reference_count']}")
        lines.append(f"- **Filtered — reference too short**: {c['reference_too_short_count']}")
        lines.append(f"- **Duplicates removed**: {c['duplicate_removed_count']}")
        lines.append(f"- **Suitability**: {c['suitability']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 3. Boundary / Less Suitable Structures")
    lines.append("")
    lines.append("These structures were scanned but are less suitable for TASD. They are kept for boundary analysis.")
    lines.append("")

    for key in info["boundary_structures"]:
        if key not in cov:
            continue
        c = cov[key]
        lines.append(f"### {c['display']} (`{key}`)")
        lines.append("")
        lines.append(f"- **Raw candidates**: {c['raw_candidate_count']}")
        lines.append(f"- **Valid candidates**: {c['valid_candidate_count']} (from {c['unique_candidate_count']} unique)")
        lines.append(f"- **Source files**: {c['source_file_count']}")
        lines.append(f"- **Repos/packages**: {c['repo_or_package_count']}")
        lines.append(f"- **Avg block lines**: {c['avg_block_lines']}")
        lines.append(f"- **Avg reference lines**: {c['avg_reference_lines']}")
        lines.append(f"- **Suitability**: {c['suitability']}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. Conclusions")
    lines.append("")
    lines.append("### Coverage Scope")
    lines.append("")
    lines.append(f"TASD targets {len(info['validated_structures'])} structure types. These represent a significant portion ")
    lines.append("of structured code completion opportunities in real Python codebases, ")
    lines.append("but TASD does **not** cover all code completion scenarios.")
    lines.append("")
    lines.append("### What TASD Covers Well")
    lines.append("")
    lines.append("- **Medium-to-high complexity structures** with repeated skeletons")
    lines.append("- **Config/pipeline/CLI blocks** where reference is long enough (5+ lines)")
    lines.append("- Structures where **Guard can detect** off-structure tokens (def/class/import)")
    lines.append("- Blocks with **clear structural boundaries** (dict/list scope)")
    lines.append("")
    lines.append("### What TASD Does NOT Cover (Boundary Report)")
    lines.append("")
    lines.append("- **Fields-type structures** (`schema_fields`, `model_fields`): Blocks are too short, ")
    lines.append("  usually single-line per field. AR baseline already performs well. ")
    lines.append("  TASD's multi-token draft advantage does not apply.")
    lines.append("- **pytest_parametrize**: Decorator structure is shallow; test values ")
    lines.append("  vary widely (strings, tuples, floats). Guard rules are error-prone ")
    lines.append("  on nested strings and expressions.")
    lines.append("")
    lines.append("### Selection Bias Safeguards")
    lines.append("")
    lines.append("- Samples were **not** selected based on TASD results")
    lines.append("- Filtering only applies to: invalid prompt seed, invalid reference, ")
    lines.append("  reference too short, duplicate blocks")
    lines.append("- No sample was removed due to low speedup, low quality, or low accept rate")
    lines.append("- Boundary structures are **reported**, not hidden")
    lines.append("- Discovery and evaluation draw from the same source pool (preliminary phase)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Coverage scan completed at model-free, static-analysis level only.*")

    md_path = os.path.join(RESULTS_DIR, "final_structure_coverage_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    data = run_coverage_scan()
    generate_md_report(data)
    print("\nDone.")
