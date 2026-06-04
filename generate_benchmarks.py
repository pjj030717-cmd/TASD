"""
Benchmark generation script for TASD.

Follows benchmark_selection_protocol.md:
- Rules-first extraction, not results-first
- Fixed random_seed=20260604 for sampling
- Tracks raw/valid candidates, duplicates, filtering reasons
- No post-hoc sample manipulation
"""
import json
import os
import random
import re
from collections import Counter

DATA_DIR = "/root/autodl-tmp/data"
os.makedirs(DATA_DIR, exist_ok=True)

SELECTION_SEED = 20260604
random.seed(SELECTION_SEED)

# Source directories for real Python files
SOURCE_DIRS = [
    "/root/miniconda3/lib/python3.12/site-packages",
    "/root/autodl-tmp/benchmark_sources/openmmlab",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmsegmentation",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmpretrain",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmengine",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmpose",
    "/root/autodl-tmp/benchmark_sources/openmmlab_mmocr",
]


def _count_lines(text):
    return len(text.strip().split("\n")) if text.strip() else 0


def _count_chars(text):
    return len(text)


def _find_python_files(dirs):
    """Find all Python files in given directories."""
    py_files = []
    for d in dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            # Skip __pycache__ and test directories
            if "__pycache__" in root or "/tests/" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
    return py_files


def _get_relative_path(filepath):
    """Get a meaningful relative path for source tracking."""
    for src in SOURCE_DIRS:
        if filepath.startswith(src):
            return filepath[len(src):].lstrip("/")
    return filepath


def _get_source_repo(filepath):
    """Determine which repo/source the file belongs to."""
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
        # Extract package name from site-packages
        rel = _get_relative_path(filepath)
        parts = rel.split("/")
        if parts:
            return f"python-{parts[0]}"
        return "python-unknown"


# ============================================================
# A. CodeSearchNet-Argparse Benchmark
# ============================================================

def extract_argparse_blocks(filepath):
    """Extract argparse add_argument blocks from a Python file."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Look for add_argument( or add_option( patterns
        if re.search(r'\.(add_argument|add_option)\s*\(', line):
            # Collect the full multi-line call
            block_start = i
            block_lines = [line]
            paren_depth = line.count("(") - line.count(")")
            j = i + 1
            while j < len(lines) and paren_depth > 0:
                block_lines.append(lines[j])
                paren_depth += lines[j].count("(") - lines[j].count(")")
                j += 1

            # Now look for ADDITIONAL add_argument calls nearby (within next 50 lines)
            # to form a complete argparse block
            additional_lines = []
            k = j
            while k < len(lines) and k < j + 50:
                if re.search(r'\.(add_argument|add_option)\s*\(', lines[k]):
                    # Found another call, collect it too
                    call_lines = [lines[k]]
                    depth = lines[k].count("(") - lines[k].count(")")
                    k += 1
                    while k < len(lines) and depth > 0:
                        call_lines.append(lines[k])
                        depth += lines[k].count("(") - lines[k].count(")")
                        k += 1
                    additional_lines.extend(call_lines)
                elif lines[k].strip() and not lines[k].strip().startswith('#'):
                    # Non-empty, non-comment line that's not add_argument - block ended
                    break
                else:
                    k += 1

            # Combine all add_argument calls
            all_block_lines = block_lines + additional_lines
            full_block = "".join(all_block_lines)
            arg_calls = re.findall(r'\.(add_argument|add_option)\s*\(', full_block)

            if len(arg_calls) >= 3:  # At least 3 arguments
                blocks.append({
                    "start_line": block_start,
                    "end_line": k - 1,
                    "lines": all_block_lines,
                    "arg_count": len(arg_calls),
                })
            i = k
        else:
            i += 1

    return blocks


def _build_argparse_sample_from_block(filepath, block, sample_idx):
    """Build a sample from a real argparse block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split into prompt (first 40-60%) and reference (rest)
    split_point = max(2, min(int(total_lines * 0.5), total_lines - 3))

    prompt_lines = all_lines[:split_point]
    ref_lines = all_lines[split_point:]

    # Ensure reference is not too short
    if len(ref_lines) < 2:
        return None

    prompt = "".join(prompt_lines).rstrip()
    reference = "".join(ref_lines).rstrip()

    # Count seed arguments in prompt
    seed_count = len(re.findall(r'\.(add_argument|add_option)\s*\(', prompt))
    ref_count = len(re.findall(r'\.(add_argument|add_option)\s*\(', reference))

    # Filter: need at least 2 seed args and reference not too short
    if seed_count < 2 or ref_count < 1:
        return None

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    return {
        "name": f"argparse_real_{sample_idx + 1:03d}",
        "source": "CodeSearchNet-Python",
        "structure_type": "argparse",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": seed_count,
            "reference_structure_count": ref_count,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
        },
    }


def generate_argparse_benchmark(target_count=80):
    """Generate argparse benchmark from real Python files."""
    print("Extracting argparse blocks from real Python files...")
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"  Found {len(py_files)} Python files")

    samples = []
    sample_idx = 0

    for filepath in py_files:
        if len(samples) >= target_count:
            break

        blocks = extract_argparse_blocks(filepath)
        for block in blocks:
            if len(samples) >= target_count:
                break

            sample = _build_argparse_sample_from_block(filepath, block, sample_idx)
            if sample:
                samples.append(sample)
                sample_idx += 1

    print(f"  Extracted {len(samples)} argparse samples")
    return samples


# ============================================================
# B. CodeSearchNet-DictConfig Benchmark
# ============================================================

def extract_dict_config_blocks(filepath):
    """Extract dict config blocks from a Python file."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return []

    blocks = []

    # Look for patterns like: var = {, var = dict(
    patterns = [
        (r'^(\w+)\s*=\s*\{', '{'),  # var = {
        (r'^(\w+)\s*=\s*dict\s*\(', 'dict('),  # var = dict(
        (r'^(\w+)\s*=\s*\[', '['),  # var = [ (list of dicts)
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern, bracket_type in patterns:
            match = re.match(pattern, line.strip())
            if match:
                var_name = match.group(1)

                # Skip if it's clearly not a config (function body, etc.)
                if var_name in ('result', 'ret', 'tmp', 'x', 'y', 'data'):
                    continue

                # Collect the block
                block_start = i
                block_lines = [line]

                if bracket_type == '{':
                    depth = line.count('{') - line.count('}')
                elif bracket_type == 'dict(':
                    depth = line.count('(') - line.count(')')
                else:  # '['
                    depth = line.count('[') - line.count(']')

                j = i + 1
                while j < len(lines) and depth > 0:
                    block_lines.append(lines[j])
                    if bracket_type == '{':
                        depth += lines[j].count('{') - lines[j].count('}')
                    elif bracket_type == 'dict(':
                        depth += lines[j].count('(') - lines[j].count(')')
                    else:
                        depth += lines[j].count('[') - lines[j].count(']')
                    j += 1

                # Check block quality
                block_text = "\n".join(block_lines)
                block_len = len(block_lines)

                # Filter: at least 8 lines, contains key-value patterns
                if block_len >= 8:
                    # Check if it looks like a config (has key: value or key= patterns)
                    has_config_pattern = bool(re.search(r'["\']\w+["\']\s*[:=]', block_text))
                    if has_config_pattern:
                        blocks.append({
                            "start_line": block_start,
                            "end_line": j - 1,
                            "lines": block_lines,
                            "var_name": var_name,
                        })

                i = j
                break
        else:
            i += 1

    return blocks


def _build_dict_config_sample_from_block(filepath, block, sample_idx):
    """Build a sample from a real dict config block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split: prompt takes first 3-6 lines, reference takes rest
    prompt_len = min(6, max(3, total_lines // 3))
    ref_len = total_lines - prompt_len

    # Ensure reference is at least 5 lines
    if ref_len < 5:
        return None

    prompt_lines = all_lines[:prompt_len]
    ref_lines = all_lines[prompt_len:]

    prompt = "\n".join(prompt_lines)
    reference = "\n".join(ref_lines)

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    return {
        "name": f"dict_config_real_{sample_idx + 1:03d}",
        "source": "CodeSearchNet-Python",
        "structure_type": "dict_config",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": prompt_len,
            "reference_structure_count": ref_len,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
        },
    }


def generate_dict_config_benchmark(target_count=80):
    """Generate dict_config benchmark from real Python files."""
    print("Extracting dict config blocks from real Python files...")
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"  Found {len(py_files)} Python files")

    samples = []
    sample_idx = 0

    for filepath in py_files:
        if len(samples) >= target_count:
            break

        blocks = extract_dict_config_blocks(filepath)
        for block in blocks:
            if len(samples) >= target_count:
                break

            sample = _build_dict_config_sample_from_block(filepath, block, sample_idx)
            if sample:
                samples.append(sample)
                sample_idx += 1

    print(f"  Extracted {len(samples)} dict_config samples")
    return samples


# ============================================================
# C. OpenMMLab-Config Benchmark
# ============================================================

def extract_openmmlab_config_blocks(filepath):
    """Extract config blocks from OpenMMLab config files."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return []

    # Skip files that are mostly imports or comments
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('import')]
    if len(code_lines) < 8:
        return []

    blocks = []

    # OpenMMLab config patterns
    patterns = [
        r'^(model|train_pipeline|test_pipeline|val_pipeline|train_dataloader|val_dataloader|test_dataloader|optim_wrapper|param_scheduler|default_hooks|custom_hooks|log_processor|env|launcher)\s*=\s*',
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                # Find the block start
                block_start = i
                block_lines = [line]

                # Determine bracket type and depth
                depth = line.count('(') - line.count(')') + line.count('[') - line.count(']') + line.count('{') - line.count('}')

                j = i + 1
                while j < len(lines) and depth > 0:
                    block_lines.append(lines[j])
                    depth += lines[j].count('(') - lines[j].count(')')
                    depth += lines[j].count('[') - lines[j].count(']')
                    depth += lines[j].count('{') - lines[j].count('}')
                    j += 1

                block_len = len(block_lines)
                if block_len >= 8:
                    blocks.append({
                        "start_line": block_start,
                        "end_line": j - 1,
                        "lines": block_lines,
                    })

                i = j
                break
        else:
            i += 1

    return blocks


def _build_openmmlab_sample_from_block(filepath, block, sample_idx):
    """Build a sample from a real OpenMMLab config block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split: prompt takes first 3-6 lines, reference takes rest
    prompt_len = min(6, max(3, total_lines // 3))
    ref_len = total_lines - prompt_len

    if ref_len < 5:
        return None

    prompt_lines = all_lines[:prompt_len]
    ref_lines = all_lines[prompt_len:]

    prompt = "\n".join(prompt_lines)
    reference = "\n".join(ref_lines)

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    return {
        "name": f"openmmlab_config_real_{sample_idx + 1:03d}",
        "source": f"OpenMMLab-{repo.split('/')[-1]}",
        "structure_type": "openmmlab_config",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": prompt_len,
            "reference_structure_count": ref_len,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
        },
    }


def generate_openmmlab_benchmark(target_count=80):
    """Generate OpenMMLab benchmark from real config files."""
    print("Extracting OpenMMLab config blocks from real config files...")

    openmmlab_dirs = [d for d in SOURCE_DIRS if "openmmlab" in d]
    
    # Collect blocks per repo for balanced sampling
    repo_blocks = {}
    for d in openmmlab_dirs:
        repo = _get_source_repo(d + "/dummy.py")
        repo_blocks[repo] = []
        
        if not os.path.exists(d):
            continue
            
        for root, _, files in os.walk(d):
            if "__pycache__" in root:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                    
                filepath = os.path.join(root, f)
                blocks = extract_openmmlab_config_blocks(filepath)
                for block in blocks:
                    block["filepath"] = filepath
                    repo_blocks[repo].append(block)
    
    print(f"  Found blocks per repo: {[(k, len(v)) for k, v in repo_blocks.items()]}")
    
    # Balanced sampling: take equal from each repo
    samples = []
    sample_idx = 0
    samples_per_repo = target_count // len([k for k, v in repo_blocks.items() if v])
    
    for repo, blocks in repo_blocks.items():
        if not blocks:
            continue
            
        for block in blocks[:samples_per_repo]:
            if len(samples) >= target_count:
                break
                
            filepath = block.pop("filepath")
            sample = _build_openmmlab_sample_from_block(filepath, block, sample_idx)
            if sample:
                samples.append(sample)
                sample_idx += 1
    
    # If we still need more samples, fill from any repo
    if len(samples) < target_count:
        for repo, blocks in repo_blocks.items():
            if len(samples) >= target_count:
                break
            for block in blocks[samples_per_repo:]:
                if len(samples) >= target_count:
                    break
                filepath = block.pop("filepath")
                sample = _build_openmmlab_sample_from_block(filepath, block, sample_idx)
                if sample:
                    samples.append(sample)
                    sample_idx += 1

    print(f"  Extracted {len(samples)} OpenMMLab samples")
    return samples


# ============================================================
# D. Pipeline-Stage-Config Benchmark
# ============================================================

def extract_pipeline_stage_blocks(filepath):
    """Extract pipeline/stage config blocks from a Python file."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    blocks = []

    # Patterns for pipeline/stage configs
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
        for pattern in patterns:
            if re.search(pattern, line):
                block_start = i
                block_lines = [line]

                # Determine bracket depth
                depth = line.count('[') - line.count(']')
                depth += line.count('(') - line.count(')')
                depth += line.count('{') - line.count('}')

                j = i + 1
                while j < len(lines) and depth > 0:
                    block_lines.append(lines[j])
                    depth += lines[j].count('[') - lines[j].count(']')
                    depth += lines[j].count('(') - lines[j].count(')')
                    depth += lines[j].count('{') - lines[j].count('}')
                    j += 1

                block_len = len(block_lines)
                if block_len >= 8:
                    # Check if it looks like a pipeline (has type= or dict patterns)
                    block_text = "".join(block_lines)
                    has_stage = bool(re.search(r'(type|name)\s*[:=]', block_text))
                    if has_stage:
                        # Count stages (number of dict/type occurrences)
                        stage_count = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', block_text))
                        blocks.append({
                            "start_line": block_start,
                            "end_line": j - 1,
                            "lines": block_lines,
                            "stage_count": stage_count,
                        })

                i = j
                break
        else:
            i += 1

    return blocks


def _build_pipeline_stage_sample(filepath, block, sample_idx):
    """Build a sample from a pipeline/stage config block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split: prompt takes first 3-6 lines, reference takes rest
    prompt_len = min(6, max(3, total_lines // 3))
    ref_len = total_lines - prompt_len

    if ref_len < 5:
        return None

    prompt_lines = all_lines[:prompt_len]
    ref_lines = all_lines[prompt_len:]

    prompt = "".join(prompt_lines).rstrip()
    reference = "".join(ref_lines).rstrip()

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    # Count seeds in prompt (stage definitions)
    prompt_text = "".join(prompt_lines)
    seed_count = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', prompt_text))
    ref_text = "".join(ref_lines)
    ref_stage_count = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', ref_text))

    # Filter: need at least 1 seed and 2 reference stages
    if seed_count < 1 or ref_stage_count < 2:
        return None

    return {
        "name": f"pipeline_stage_config_{sample_idx + 1:03d}",
        "source": "Pipeline-Stage-Config",
        "structure_type": "pipeline_stage_config",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": seed_count,
            "reference_structure_count": ref_stage_count,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
            "nesting_depth": 0,
            "augmented_from_same_block": False,
        },
    }


def generate_pipeline_stage_benchmark(target_count=20):
    """Generate pipeline stage config benchmark (protocol-compliant)."""
    print("Extracting pipeline stage config blocks...")
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"  Found {len(py_files)} Python files")

    all_candidates = []
    seen_blocks = set()

    # Phase 1: Extract all raw candidates
    raw_candidate_count = 0
    for filepath in py_files:
        blocks = extract_pipeline_stage_blocks(filepath)
        for block in blocks:
            raw_candidate_count += 1
            block_key = (filepath, block["start_line"])
            if block_key in seen_blocks:
                continue
            seen_blocks.add(block_key)
            all_candidates.append((filepath, block))

    print(f"  Raw candidates extracted: {raw_candidate_count}")
    print(f"  Unique candidates (after dedup): {len(all_candidates)}")

    duplicate_removed_count = raw_candidate_count - len(all_candidates)

    # Phase 2: Filter into valid candidates
    valid_candidates = []
    invalid_seed_count = 0
    invalid_ref_count = 0
    ref_too_short_count = 0

    for filepath, block in all_candidates:
        sample = _build_pipeline_stage_sample(filepath, block, 0)
        if sample is None:
            # Determine filter reason
            all_lines = block["lines"]
            prompt_text = "".join(all_lines[:min(6, max(3, len(all_lines)//3))])
            seed_count = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', prompt_text))
            ref_text = "".join(all_lines[min(6, max(3, len(all_lines)//3)):])
            ref_stage_count = len(re.findall(r'(?:dict\s*\(\s*type|["\']type["\']\s*:)', ref_text))

            if seed_count < 1:
                invalid_seed_count += 1
            elif ref_stage_count < 2:
                invalid_ref_count += 1
            else:
                ref_too_short_count += 1
            continue

        valid_candidates.append((filepath, block, sample))

    print(f"  Valid candidates: {len(valid_candidates)}")
    print(f"    invalid_prompt_seed: {invalid_seed_count}")
    print(f"    invalid_reference: {invalid_ref_count}")
    print(f"    reference_too_short: {ref_too_short_count}")

    # Phase 3: Shuffle with fixed seed and sample
    random.shuffle(valid_candidates)
    selected = valid_candidates[:target_count]

    samples = []
    for sample_idx, (filepath, block, sample) in enumerate(selected):
        sample["name"] = f"pipeline_stage_config_{sample_idx + 1:03d}"
        samples.append(sample)

    print(f"  Selected {len(samples)} samples (seed={SELECTION_SEED})")

    stats = {
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": len(valid_candidates),
        "duplicate_removed_count": duplicate_removed_count,
        "reference_too_short_count": ref_too_short_count,
        "invalid_prompt_seed_count": invalid_seed_count,
        "invalid_reference_count": invalid_ref_count,
        "notes": [
            f"Phase: pilot (20 samples)"
        ],
    }

    return samples, stats


# ============================================================
# E. Complex-Nested-Config Benchmark
# ============================================================

def extract_complex_nested_config_blocks(filepath):
    """Extract complex nested config blocks from a Python file."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    blocks = []

    # Patterns for complex nested configs
    patterns = [
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG)\s*=\s*\{', '{'),
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG)\s*=\s*dict\s*\(', 'dict('),
        (r'^\s*dict\s*\(\s*\w+\s*=\s*dict\s*\(', 'nested_dict'),
        (r'^\s*\{\s*["\']\w+["\']\s*:\s*\{', 'nested_literal'),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern, bracket_type in patterns:
            if re.search(pattern, line.strip()):
                block_start = i
                block_lines = [line]

                if bracket_type == '{':
                    depth = line.count('{') - line.count('}')
                elif bracket_type == 'dict(':
                    depth = line.count('(') - line.count(')')
                else:
                    depth = line.count('{') - line.count('}')
                    depth += line.count('(') - line.count(')')

                j = i + 1
                while j < len(lines) and depth > 0:
                    block_lines.append(lines[j])
                    if bracket_type == '{' or bracket_type in ('nested_dict', 'nested_literal'):
                        depth += lines[j].count('{') - lines[j].count('}')
                    if bracket_type in ('dict(', 'nested_dict'):
                        depth += lines[j].count('(') - lines[j].count(')')
                    j += 1

                block_len = len(block_lines)
                if block_len >= 8:
                    block_text = "".join(block_lines)
                    # Check nesting depth (at least 2 levels)
                    max_depth = 0
                    current = 0
                    for ch in block_text:
                        if ch in '{(':
                            current += 1
                            max_depth = max(max_depth, current)
                        elif ch in '})':
                            current -= 1

                    if max_depth >= 3:  # At least 2 levels of nesting
                        blocks.append({
                            "start_line": block_start,
                            "end_line": j - 1,
                            "lines": block_lines,
                            "nesting_depth": max_depth,
                        })

                i = j
                break
        else:
            i += 1

    return blocks


def extract_complex_nested_config_blocks_v2(filepath):
    """Extract complex nested config blocks - relaxed version."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    blocks = []

    # More patterns for complex nested configs
    patterns = [
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG|options|OPTIONS|args|ARGS)\s*=\s*\{', '{'),
        (r'^(config|CONFIG|DEFAULTS|settings|params|cfg|CFG|options|OPTIONS|args|ARGS)\s*=\s*dict\s*\(', 'dict('),
        (r'^\s*dict\s*\(\s*\w+\s*=\s*dict\s*\(', 'nested_dict'),
        (r'^\s*\{\s*["\']\w+["\']\s*:\s*\{', 'nested_literal'),
        (r'^\s*["\']\w+["\']\s*:\s*\{', 'inline_nested'),
        (r'^\s*\w+\s*=\s*\{', 'assignment_dict'),
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern, bracket_type in patterns:
            if re.search(pattern, line.strip()):
                block_start = i
                block_lines = [line]

                if bracket_type == '{':
                    depth = line.count('{') - line.count('}')
                elif bracket_type == 'dict(':
                    depth = line.count('(') - line.count(')')
                else:
                    depth = line.count('{') - line.count('}')
                    depth += line.count('(') - line.count(')')

                j = i + 1
                while j < len(lines) and depth > 0:
                    block_lines.append(lines[j])
                    if bracket_type == '{' or bracket_type in ('nested_dict', 'nested_literal', 'inline_nested', 'assignment_dict'):
                        depth += lines[j].count('{') - lines[j].count('}')
                    if bracket_type in ('dict(', 'nested_dict'):
                        depth += lines[j].count('(') - lines[j].count(')')
                    j += 1

                block_len = len(block_lines)
                if block_len >= 8:
                    block_text = "".join(block_lines)
                    # Check nesting depth (at least 2 levels)
                    max_depth = 0
                    current = 0
                    for ch in block_text:
                        if ch in '{(':
                            current += 1
                            max_depth = max(max_depth, current)
                        elif ch in '})':
                            current -= 1

                    if max_depth >= 3:  # At least 2 levels of nesting
                        blocks.append({
                            "start_line": block_start,
                            "end_line": j - 1,
                            "lines": block_lines,
                            "nesting_depth": max_depth,
                        })

                i = j
                break
        else:
            i += 1

    return blocks


def _build_complex_nested_sample(filepath, block, sample_idx):
    """Build a sample from a complex nested config block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split: prompt takes first 4-8 lines, reference takes rest
    prompt_len = min(8, max(4, total_lines // 3))
    ref_len = total_lines - prompt_len

    if ref_len < 8:
        return None

    prompt_lines = all_lines[:prompt_len]
    ref_lines = all_lines[prompt_len:]

    prompt = "\n".join(prompt_lines).rstrip()
    reference = "\n".join(ref_lines).rstrip()

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    return {
        "name": f"complex_nested_config_{sample_idx + 1:03d}",
        "source": "Complex-Nested-Config",
        "structure_type": "complex_nested_config",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": prompt_len,
            "reference_structure_count": ref_len,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
            "nesting_depth": block["nesting_depth"],
            "augmented_from_same_block": False,
        },
    }


def generate_complex_nested_benchmark(target_count=20):
    """Generate complex nested config benchmark (protocol-compliant)."""
    print("Extracting complex nested config blocks...")
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"  Found {len(py_files)} Python files")

    all_candidates = []
    seen_blocks = set()
    raw_candidate_count = 0

    # Phase 1: strict extractor
    for filepath in py_files:
        blocks = extract_complex_nested_config_blocks(filepath)
        for block in blocks:
            raw_candidate_count += 1
            block_key = (filepath, block["start_line"])
            if block_key in seen_blocks:
                continue
            seen_blocks.add(block_key)
            all_candidates.append((filepath, block))

    # Phase 1b: relaxed extractor
    for filepath in py_files:
        blocks = extract_complex_nested_config_blocks_v2(filepath)
        for block in blocks:
            raw_candidate_count += 1
            block_key = (filepath, block["start_line"])
            if block_key in seen_blocks:
                continue
            seen_blocks.add(block_key)
            all_candidates.append((filepath, block))

    print(f"  Raw candidates extracted: {raw_candidate_count}")
    print(f"  Unique candidates (after dedup): {len(all_candidates)}")

    duplicate_removed_count = raw_candidate_count - len(all_candidates)

    # Phase 2: Filter into valid candidates
    valid_candidates = []
    invalid_seed_count = 0
    invalid_ref_count = 0
    ref_too_short_count = 0

    for filepath, block in all_candidates:
        sample = _build_complex_nested_sample(filepath, block, 0)
        if sample is None:
            all_lines = block["lines"]
            ref_len = len(all_lines) - min(8, max(4, len(all_lines)//3))
            if ref_len < 8:
                ref_too_short_count += 1
            else:
                invalid_ref_count += 1
            continue

        valid_candidates.append((filepath, block, sample))

    print(f"  Valid candidates: {len(valid_candidates)}")
    print(f"    invalid_prompt_seed: {invalid_seed_count}")
    print(f"    invalid_reference: {invalid_ref_count}")
    print(f"    reference_too_short: {ref_too_short_count}")

    # Phase 3: Shuffle with fixed seed and sample
    random.shuffle(valid_candidates)
    selected = valid_candidates[:target_count]

    samples = []
    for sample_idx, (filepath, block, sample) in enumerate(selected):
        sample["name"] = f"complex_nested_config_{sample_idx + 1:03d}"
        samples.append(sample)

    print(f"  Selected {len(samples)} samples (seed={SELECTION_SEED})")

    stats = {
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": len(valid_candidates),
        "duplicate_removed_count": duplicate_removed_count,
        "reference_too_short_count": ref_too_short_count,
        "invalid_prompt_seed_count": invalid_seed_count,
        "invalid_reference_count": invalid_ref_count,
        "notes": [
            f"Phase: pilot (20 samples)",
            f"Used strict + relaxed extractors"
        ],
    }

    return samples, stats


# ============================================================
# F. Rich-CLI-Option-Groups Benchmark
# ============================================================

def extract_rich_cli_option_blocks(filepath):
    """Extract rich CLI option groups from a Python file."""
    try:
        with open(filepath, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return []

    blocks = []

    # Patterns for rich CLI options
    patterns = [
        r'\.(add_argument|add_option)\s*\(',
        r'click\.option\s*\(',
        r'typer\.Option\s*\(',
        r'arg\s*=\s*\[',
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern in patterns:
            if re.search(pattern, line):
                block_start = i
                all_option_lines = []

                # Collect all nearby option definitions
                j = i
                while j < len(lines) and j < i + 100:
                    if re.search(pattern, lines[j]):
                        # Collect this multi-line call
                        call_lines = [lines[j]]
                        depth = lines[j].count('(') - lines[j].count(')')
                        depth += lines[j].count('[') - lines[j].count(']')
                        k = j + 1
                        while k < len(lines) and depth > 0:
                            call_lines.append(lines[k])
                            depth += lines[k].count('(') - lines[k].count(')')
                            depth += lines[k].count('[') - lines[k].count(']')
                            k += 1
                        all_option_lines.extend(call_lines)
                        j = k
                    elif lines[j].strip() and not lines[j].strip().startswith('#'):
                        break
                    else:
                        j += 1

                # Count options and check for rich fields
                option_count = len(re.findall(pattern, "".join(all_option_lines)))
                block_text = "".join(all_option_lines)
                has_rich_fields = bool(re.search(r'(choices|default|type|action|help|required|nargs|metavar)\s*=', block_text))

                if option_count >= 6 and has_rich_fields:
                    blocks.append({
                        "start_line": block_start,
                        "end_line": j - 1,
                        "lines": all_option_lines,
                        "option_count": option_count,
                    })

                i = j
                break
        else:
            i += 1

    return blocks


def _build_rich_cli_sample(filepath, block, sample_idx):
    """Build a sample from a rich CLI option block."""
    all_lines = block["lines"]
    total_lines = len(all_lines)

    # Split: prompt takes first 2-3 options, reference takes rest
    # Find split point after 2-3 complete options
    option_pattern = r'\.(add_argument|add_option)\s*\(|click\.option\s*\(|typer\.Option\s*\('
    option_positions = []
    for idx, line in enumerate(all_lines):
        if re.search(option_pattern, line):
            option_positions.append(idx)

    if len(option_positions) < 3:
        return None

    # Split after 2-3 options
    split_idx = option_positions[min(2, len(option_positions) - 2)]

    prompt_lines = all_lines[:split_idx]
    ref_lines = all_lines[split_idx:]

    if len(ref_lines) < 3:
        return None

    prompt = "".join(prompt_lines).rstrip()
    reference = "".join(ref_lines).rstrip()

    rel_path = _get_relative_path(filepath)
    repo = _get_source_repo(filepath)

    prompt_text = "".join(prompt_lines)
    seed_count = len(re.findall(option_pattern, prompt_text))
    ref_text = "".join(ref_lines)
    ref_option_count = len(re.findall(option_pattern, ref_text))

    return {
        "name": f"rich_cli_option_groups_{sample_idx + 1:03d}",
        "source": "Rich-CLI-Option-Groups",
        "structure_type": "rich_cli_option_groups",
        "prompt": prompt,
        "reference": reference,
        "metadata": {
            "source_repo": repo,
            "source_file": rel_path,
            "seed_count": seed_count,
            "reference_structure_count": ref_option_count,
            "block_start_line": block["start_line"],
            "block_end_line": block["end_line"],
            "nesting_depth": 0,
            "augmented_from_same_block": False,
        },
    }


def generate_rich_cli_benchmark(target_count=20):
    """Generate rich CLI option groups benchmark (protocol-compliant)."""
    print("Extracting rich CLI option groups...")
    py_files = _find_python_files(SOURCE_DIRS)
    print(f"  Found {len(py_files)} Python files")

    all_candidates = []
    seen_blocks = set()
    raw_candidate_count = 0

    # Phase 1: Extract all raw candidates
    for filepath in py_files:
        blocks = extract_rich_cli_option_blocks(filepath)
        for block in blocks:
            raw_candidate_count += 1
            block_key = (filepath, block["start_line"])
            if block_key in seen_blocks:
                continue
            seen_blocks.add(block_key)
            all_candidates.append((filepath, block))

    print(f"  Raw candidates extracted: {raw_candidate_count}")
    print(f"  Unique candidates (after dedup): {len(all_candidates)}")

    duplicate_removed_count = raw_candidate_count - len(all_candidates)

    # Phase 2: Filter into valid candidates
    valid_candidates = []
    invalid_seed_count = 0
    invalid_ref_count = 0
    ref_too_short_count = 0

    option_pattern = r'\.(add_argument|add_option)\s*\(|click\.option\s*\(|typer\.Option\s*\('

    for filepath, block in all_candidates:
        sample = _build_rich_cli_sample(filepath, block, 0)
        if sample is None:
            # Determine filter reason
            option_positions = []
            for idx, line in enumerate(block["lines"]):
                if re.search(option_pattern, line):
                    option_positions.append(idx)

            if len(option_positions) < 3:
                invalid_seed_count += 1
            else:
                split_idx = option_positions[min(2, len(option_positions) - 2)]
                ref_lines = block["lines"][split_idx:]
                if len(ref_lines) < 3:
                    ref_too_short_count += 1
                else:
                    invalid_ref_count += 1
            continue

        valid_candidates.append((filepath, block, sample))

    print(f"  Valid candidates: {len(valid_candidates)}")
    print(f"    invalid_prompt_seed: {invalid_seed_count}")
    print(f"    invalid_reference: {invalid_ref_count}")
    print(f"    reference_too_short: {ref_too_short_count}")

    # Phase 3: Shuffle with fixed seed and sample
    random.shuffle(valid_candidates)
    selected = valid_candidates[:target_count]

    samples = []
    for sample_idx, (filepath, block, sample) in enumerate(selected):
        sample["name"] = f"rich_cli_option_groups_{sample_idx + 1:03d}"
        samples.append(sample)

    print(f"  Selected {len(samples)} samples (seed={SELECTION_SEED})")

    stats = {
        "raw_candidate_count": raw_candidate_count,
        "valid_candidate_count": len(valid_candidates),
        "duplicate_removed_count": duplicate_removed_count,
        "reference_too_short_count": ref_too_short_count,
        "invalid_prompt_seed_count": invalid_seed_count,
        "invalid_reference_count": invalid_ref_count,
        "notes": [
            f"Phase: pilot (20 samples)"
        ],
    }

    return samples, stats


# ============================================================
# Write benchmark files
# ============================================================

def write_benchmark(samples, basename, stats_override=None):
    """Write benchmark JSONL and summary. Accepts optional stats_override dict with
    protocol tracking fields like raw_candidate_count, valid_candidate_count, etc."""
    jsonl_path = os.path.join(DATA_DIR, f"{basename}.jsonl")
    summary_path = os.path.join(DATA_DIR, f"{basename}_summary.json")

    with open(jsonl_path, "w") as f:
        for s in samples:
            prompt_lines = _count_lines(s["prompt"])
            ref_lines = _count_lines(s["reference"])
            prompt_chars = _count_chars(s["prompt"])
            ref_chars = _count_chars(s["reference"])

            record = {
                "name": s["name"],
                "source": s["source"],
                "structure_type": s["structure_type"],
                "prompt": s["prompt"],
                "reference": s["reference"],
                "metadata": {
                    **s["metadata"],
                    "prompt_lines": prompt_lines,
                    "reference_lines": ref_lines,
                    "prompt_chars": prompt_chars,
                    "reference_chars": ref_chars,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    source_counts = dict(Counter(s["source"] for s in samples))
    repo_counts = dict(Counter(s["metadata"]["source_repo"] for s in samples))
    source_file_count = len(set(s["metadata"]["source_file"] for s in samples))
    repo_count = len(repo_counts)
    avg_prompt_lines = sum(_count_lines(s["prompt"]) for s in samples) / max(len(samples), 1)
    avg_ref_lines = sum(_count_lines(s["reference"]) for s in samples) / max(len(samples), 1)
    avg_prompt_chars = sum(_count_chars(s["prompt"]) for s in samples) / max(len(samples), 1)
    avg_ref_chars = sum(_count_chars(s["reference"]) for s in samples) / max(len(samples), 1)

    min_prompt_chars = min((_count_chars(s["prompt"]) for s in samples), default=0)
    max_prompt_chars = max((_count_chars(s["prompt"]) for s in samples), default=0)
    min_ref_chars = min((_count_chars(s["reference"]) for s in samples), default=0)
    max_ref_chars = max((_count_chars(s["reference"]) for s in samples), default=0)

    invalid_seed = sum(1 for s in samples if s["metadata"].get("seed_count", 0) < 2)
    invalid_ref = sum(1 for s in samples if _count_lines(s["reference"]) < 5)

    notes = [f"Extracted {len(samples)} real samples from source code for {basename}"]
    if stats_override:
        notes.extend(stats_override.get("notes", []))
        raw_seed_invalid = stats_override.get("invalid_prompt_seed_count", invalid_seed)
        raw_ref_invalid = stats_override.get("invalid_reference_count", invalid_ref)
    else:
        raw_seed_invalid = invalid_seed
        raw_ref_invalid = invalid_ref

    summary = {
        "sample_count": len(samples),
        "raw_candidate_count": stats_override.get("raw_candidate_count", len(samples)) if stats_override else len(samples),
        "valid_candidate_count": stats_override.get("valid_candidate_count", len(samples)) if stats_override else len(samples),
        "selected_sample_count": len(samples),
        "source_file_count": source_file_count,
        "repo_count": repo_count,
        "source_counts": source_counts,
        "repo_counts": repo_counts,
        "avg_prompt_lines": round(avg_prompt_lines, 2),
        "avg_reference_lines": round(avg_ref_lines, 2),
        "avg_prompt_chars": round(avg_prompt_chars, 2),
        "avg_reference_chars": round(avg_ref_chars, 2),
        "min_prompt_chars": min_prompt_chars,
        "max_prompt_chars": max_prompt_chars,
        "min_reference_chars": min_ref_chars,
        "max_reference_chars": max_ref_chars,
        "invalid_prompt_seed_count": raw_seed_invalid,
        "invalid_reference_count": raw_ref_invalid,
        "reference_too_short_count": stats_override.get("reference_too_short_count", 0) if stats_override else 0,
        "duplicate_removed_count": stats_override.get("duplicate_removed_count", 0) if stats_override else 0,
        "selection_seed": SELECTION_SEED,
        "notes": notes,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"  Written {jsonl_path} ({len(samples)} samples)")
    print(f"  Written {summary_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    random.seed(SELECTION_SEED)
    print("Generating new benchmarks for TASD evaluation...")
    print(f"Selection seed: {SELECTION_SEED}")

    print("\n=== D. Pipeline-Stage-Config (20 samples) ===")
    pipeline_samples, pipeline_stats = generate_pipeline_stage_benchmark(20)
    write_benchmark(pipeline_samples, "pipeline_stage_config_20", pipeline_stats)

    print("\n=== E. Complex-Nested-Config (20 samples) ===")
    nested_samples, nested_stats = generate_complex_nested_benchmark(20)
    write_benchmark(nested_samples, "complex_nested_config_20", nested_stats)

    print("\n=== F. Rich-CLI-Option-Groups (20 samples) ===")
    cli_samples, cli_stats = generate_rich_cli_benchmark(20)
    write_benchmark(cli_samples, "rich_cli_option_groups_20", cli_stats)

    print("\nDone.")
