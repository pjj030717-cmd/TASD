# TASD 480-Sample Hard-Case Analysis

## 1. Global Statistics

- **Total samples**: 480
- **Hard cases**: 79 (16.5%)
  - Performance hard cases (low speedup/accept/high repair): 24 (5.0%)
  - Quality-flagged cases (good speedup but low SQ/off-structure): 55 (11.5%)
- **Strong cases**: 171 (35.6%)

### Per-Benchmark Statistics

| Benchmark | Total | Hard Cases | Hard Rate | Perf Hard | Quality Flagged | Strong Cases | Strong Rate | Mean Speedup | Median Speedup | Mean Accept | Mean Repair | Mean SQ |
|-----------|-------|------------|-----------|-----------|-----------------|--------------|-------------|--------------|----------------|-------------|-------------|---------|
| Real-Python-Argparse | 80 | 11 | 13.8% | 7 | 4 | 33 | 41.2% | 1.87x | 2.00x | 0.93 | 0.6 | 0.901 |
| Real-Python-DictConfig | 80 | 15 | 18.8% | 11 | 4 | 15 | 18.8% | 1.80x | 1.96x | 0.91 | 0.3 | 0.832 |
| OpenMMLab-Config | 80 | 15 | 18.8% | 6 | 9 | 37 | 46.3% | 1.93x | 2.01x | 0.95 | 0.3 | 0.888 |
| Rich-CLI-Option-Groups | 80 | 10 | 12.5% | 0 | 10 | 35 | 43.8% | 2.01x | 2.01x | 1.00 | 0.0 | 0.912 |
| Complex-Nested-Config | 80 | 20 | 25.0% | 0 | 20 | 15 | 18.8% | 2.00x | 1.99x | 1.00 | 0.0 | 0.823 |
| Pipeline-Stage-Config | 80 | 8 | 10.0% | 0 | 8 | 36 | 45.0% | 2.00x | 2.00x | 1.00 | 0.0 | 0.941 |

## 2. Worst 30 Hard Cases (by TASD Speedup)

| # | Benchmark | Sample | Source | AR TPS | TASD TPS | Speedup | Accept | Repair | SQ | OffStr | Trunc | Reasons |
|---|-----------|--------|--------|--------|----------|---------|--------|--------|----|--------|-------|---------|
| 1 | Real-Python-Argparse | 22 | conda/cli/main_run.py | 34.0 | 8.1 | 0.24x | 0.05 | 14 | 0.717 | 0.114 | 0.000 | A, B, D, F, G |
| 2 | Real-Python-Argparse | 73 | accelerate/commands/launch.py | 33.5 | 8.7 | 0.26x | 0.12 | 15 | 0.906 | 0.000 | 0.000 | A, B, D |
| 3 | Real-Python-DictConfig | 59 | pip/_vendor/typing_extensions.py | 34.2 | 11.9 | 0.35x | 0.17 | 0 | 0.714 | 0.000 | 0.000 | A |
| 4 | Real-Python-DictConfig | 2 | typing_extensions.py | 32.1 | 11.2 | 0.35x | 0.17 | 0 | 0.714 | 0.000 | 0.000 | A |
| 5 | Real-Python-DictConfig | 1 | ipython_pygments_lexers.py | 31.7 | 15.6 | 0.49x | 0.24 | 0 | 0.675 | 0.000 | 0.000 | A, C, D |
| 6 | OpenMMLab-Config | 70 | _mmocr/configs/textdet/_base_/pretrain_runtime.py | 33.5 | 16.6 | 0.50x | 0.24 | 7 | 0.685 | 0.118 | 0.118 | A, B, G, H |
| 7 | OpenMMLab-Config | 64 | _mmocr/configs/kie/_base_/default_runtime.py | 31.5 | 19.6 | 0.62x | 0.28 | 5 | 0.722 | 0.000 | 0.111 | A, B, H |
| 8 | Real-Python-Argparse | 33 | pip/_internal/commands/freeze.py | 32.6 | 20.3 | 0.62x | 0.29 | 5 | 0.794 | 0.000 | 0.375 | A, B, D, F, H |
| 9 | OpenMMLab-Config | 48 | _mmpose/configs/_base_/default_runtime.py | 33.8 | 22.7 | 0.67x | 0.34 | 4 | 0.820 | 0.100 | 0.000 | A, B, G |
| 10 | Real-Python-Argparse | 30 | pip/_internal/commands/completion.py | 32.7 | 23.6 | 0.72x | 0.34 | 4 | 0.850 | 0.000 | 0.000 | A, B, F |
| 11 | Real-Python-Argparse | 61 | accelerate/commands/launch.py | 32.1 | 23.2 | 0.72x | 0.33 | 4 | 0.829 | 0.000 | 0.000 | A, B |
| 12 | OpenMMLab-Config | 0 | configs/_base_/datasets/ade20k_instance.py | 34.2 | 27.6 | 0.81x | 0.41 | 3 | 0.860 | 0.100 | 0.100 | A, B, D, G |
| 13 | OpenMMLab-Config | 2 | configs/_base_/datasets/ade20k_panoptic.py | 33.4 | 27.4 | 0.82x | 0.41 | 3 | 0.860 | 0.100 | 0.100 | A, B, D, G |
| 14 | Real-Python-DictConfig | 13 | conda/exception_handler.py | 34.4 | 29.6 | 0.86x | 0.43 | 3 | 0.858 | 0.000 | 0.105 | A, B, H |
| 15 | Real-Python-Argparse | 69 | accelerate/commands/launch.py | 32.1 | 29.3 | 0.91x | 0.44 | 2 | 0.725 | 0.000 | 0.250 | A, F, H |
| 16 | Real-Python-DictConfig | 18 | conda/_vendor/cpuinfo/cpuinfo.py | 34.2 | 32.8 | 0.96x | 0.50 | 4 | 0.881 | 0.000 | 0.062 | A, B |
| 17 | Real-Python-Argparse | 38 | pip/_internal/commands/install.py | 34.5 | 36.8 | 1.07x | 0.54 | 2 | 0.825 | 0.000 | 0.250 | A, H |
| 18 | Real-Python-DictConfig | 57 | pip/_internal/index/collector.py | 34.3 | 41.4 | 1.21x | 0.73 | 3 | 0.680 | 0.000 | 0.300 | B, C, D, H |
| 19 | Real-Python-DictConfig | 40 | conda/gateways/repodata/jlap/fetch.py | 32.2 | 39.2 | 1.22x | 0.63 | 1 | 0.677 | 0.000 | 0.136 | A, H |
| 20 | Real-Python-DictConfig | 52 | menuinst/platforms/win_utils/knownfolders.py | 34.8 | 43.8 | 1.26x | 0.69 | 1 | 0.839 | 0.000 | 0.111 | A, D, H |
| 21 | Real-Python-DictConfig | 77 | pip/_vendor/pygments/lexers/python.py | 32.8 | 42.9 | 1.30x | 0.68 | 3 | 0.950 | 0.000 | 0.200 | A, B, D, F, H |
| 22 | OpenMMLab-Config | 29 | _mmsegmentation/configs/_base_/datasets/chase_db1.py | 31.6 | 41.7 | 1.32x | 0.70 | 4 | 0.741 | 0.000 | 0.091 | A, B, D |
| 23 | Real-Python-DictConfig | 78 | pip/_vendor/pygments/lexers/python.py | 32.2 | 42.7 | 1.33x | 0.68 | 3 | 0.950 | 0.000 | 0.200 | A, B, D, F, H |
| 24 | Real-Python-DictConfig | 17 | conda/utils.py | 34.6 | 47.8 | 1.38x | 0.71 | 1 | 0.888 | 0.000 | 0.040 |  |
| 25 | Real-Python-Argparse | 77 | accelerate/commands/to_fsdp2.py | 33.4 | 47.3 | 1.42x | 0.73 | 1 | 0.874 | 0.000 | 0.053 |  |
| 26 | Real-Python-DictConfig | 7 | cffi/backend_ctypes.py | 32.4 | 46.2 | 1.43x | 0.68 | 1 | 0.879 | 0.000 | 0.071 | A, F |
| 27 | Rich-CLI-Option-Groups | 41 | pip/_internal/commands/wheel.py | 33.8 | 48.5 | 1.44x | 0.72 | 1 | 0.989 | 0.000 | 0.111 |  |
| 28 | Real-Python-DictConfig | 43 | conda/testing/helpers.py | 32.1 | 47.9 | 1.49x | 0.74 | 1 | 0.850 | 0.000 | 0.000 |  |
| 29 | Pipeline-Stage-Config | 18 | configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py | 33.4 | 50.7 | 1.52x | 0.74 | 1 | 0.756 | 0.333 | 0.444 | D, G, H |
| 30 | Real-Python-DictConfig | 20 | conda/_vendor/cpuinfo/cpuinfo.py | 34.0 | 52.2 | 1.54x | 0.81 | 1 | 0.821 | 0.000 | 0.286 |  |

## 3. Benchmark-Level Analysis

### Real-Python-Argparse

- Hard cases: 11/80 (13.8%)
  - Performance hard: 7 (8.7%)
  - Quality-flagged: 4 (5.0%)
- Mean speedup: 1.87x, Median: 2.00x
- Mean accept rate: 0.93
- Mean repair count: 0.6
- Mean SQ: 0.901
- Main failure reasons: A (8), D (6), F (6)
- Worst 5 perf-hard avg speedup: 0.51x
- Concentrated in few samples: Yes
- Low accept (<0.70): 7/11
- High repair (>=3): 5/11
- Low SQ (<0.75): 6/11
- High off-structure (>0.05): 1/11

### Real-Python-DictConfig

- Hard cases: 15/80 (18.8%)
  - Performance hard: 11 (13.8%)
  - Quality-flagged: 4 (5.0%)
- Mean speedup: 1.80x, Median: 1.96x
- Mean accept rate: 0.91
- Mean repair count: 0.3
- Mean SQ: 0.832
- Main failure reasons: A (11), H (9), D (8)
- Worst 5 perf-hard avg speedup: 0.60x
- Concentrated in few samples: Yes
- Low accept (<0.70): 10/15
- High repair (>=3): 5/15
- Low SQ (<0.75): 9/15
- High off-structure (>0.05): 0/15

### OpenMMLab-Config

- Hard cases: 15/80 (18.8%)
  - Performance hard: 6 (7.5%)
  - Quality-flagged: 9 (11.3%)
- Mean speedup: 1.93x, Median: 2.01x
- Mean accept rate: 0.95
- Mean repair count: 0.3
- Mean SQ: 0.888
- Main failure reasons: D (12), A (6), B (6)
- Worst 5 perf-hard avg speedup: 0.68x
- Concentrated in few samples: Yes
- Low accept (<0.70): 6/15
- High repair (>=3): 6/15
- Low SQ (<0.75): 12/15
- High off-structure (>0.05): 5/15

### Rich-CLI-Option-Groups

- Hard cases: 10/80 (12.5%)
  - Performance hard: 0 (0.0%)
  - Quality-flagged: 10 (12.5%)
- Mean speedup: 2.01x, Median: 2.01x
- Mean accept rate: 1.00
- Mean repair count: 0.0
- Mean SQ: 0.912
- Main failure reasons: G (10), D (8), F (8)
- Low accept (<0.70): 0/10
- High repair (>=3): 0/10
- Low SQ (<0.75): 10/10
- High off-structure (>0.05): 10/10

### Complex-Nested-Config

- Hard cases: 20/80 (25.0%)
  - Performance hard: 0 (0.0%)
  - Quality-flagged: 20 (25.0%)
- Mean speedup: 2.00x, Median: 1.99x
- Mean accept rate: 1.00
- Mean repair count: 0.0
- Mean SQ: 0.823
- Main failure reasons: E (20), D (11), F (11)
- Low accept (<0.70): 0/20
- High repair (>=3): 0/20
- Low SQ (<0.75): 15/20
- High off-structure (>0.05): 10/20

### Pipeline-Stage-Config

- Hard cases: 8/80 (10.0%)
  - Performance hard: 0 (0.0%)
  - Quality-flagged: 8 (10.0%)
- Mean speedup: 2.00x, Median: 2.00x
- Mean accept rate: 1.00
- Mean repair count: 0.0
- Mean SQ: 0.941
- Main failure reasons: D (8), G (8), H (6)
- Low accept (<0.70): 0/8
- High repair (>=3): 0/8
- Low SQ (<0.75): 0/8
- High off-structure (>0.05): 8/8

## 4. Failure Reason Classification

| Code | Reason | Count | % of Hard Cases |
|------|--------|-------|-----------------|
| A | Low accept / draft-target divergence | 25 | 31.6% |
| B | Repeated repair overhead | 16 | 20.3% |
| C | Comment or natural-language text | 2 | 2.5% |
| D | Path / filename / dataset-specific string | 53 | 67.1% |
| E | Deep nested structure | 20 | 25.3% |
| F | Unstable key/order divergence | 29 | 36.7% |
| G | Off-structure transition | 34 | 43.0% |
| H | Truncation / long unclosed bracket | 33 | 41.8% |
| I | Timing noise / short generated length | 0 | 0.0% |

## 5. Text Examples (3 Worst per Benchmark)

### Real-Python-Argparse

#### Sample 22: argparse_real_023 (speedup: 0.24x)

- Source: conda/cli/main_run.py
- Accept: 0.05, Repair: 14, SQ: 0.717, OffStr: 0.114
- Failure reasons: A, B, D, F, G

**Prompt (first 20 lines):**
```
    p.add_argument(
        "--dev",
        action=NullCountAction,
        help="Sets `CONDA_EXE` to `python -m conda`, assuming the current "
        "working directory contains the root of conda development sources. "
        "This is mainly for use during tests where we test new conda sources "
        "against old Python versions.",
        dest="dev",
        default=NULL,
    )
    p.add_argument(
        "--debug-wrapper-scripts",
        action=NullCountAction,
        help="When this is set, where implemented, the shell wrapper scripts"
        "will use the echo command to print debugging information to "
        "stderr (standard error).",
        dest="debug_wrapper_scripts",
        default=NULL,
    )
```

**Reference (first 20 lines):**
```
    p.add_argument(
        "--cwd",
        help="Current working directory for command to run in. Defaults to "
        "the user's current working directory if no directory is specified.",
        default=os.getcwd(),
    )
    p.add_argument(
        "--no-capture-output",
        "--live-stream",
        action="store_true",
        help="Don't capture stdout/stderr (standard out/standard error).",
        default=False,
    )
    p.add_argument(
        "executable_call",
        nargs=REMAINDER,
        help="Executable name, with additional arguments to be passed to the executable "
        "on invocation.",
    )
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

#### Sample 73: argparse_real_074 (speedup: 0.26x)

- Source: accelerate/commands/launch.py
- Accept: 0.12, Repair: 15, SQ: 0.906, OffStr: 0.000
- Failure reasons: A, B, D

**Prompt (first 20 lines):**
```
    aws_args.add_argument(
        "--aws_access_key_id",
        type=str,
        default=None,
        help="The AWS_ACCESS_KEY_ID used to launch the Amazon SageMaker training job",
    )
    aws_args.add_argument(
        "--aws_secret_access_key",
        type=str,
        default=None,
        help="The AWS_SECRET_ACCESS_KEY used to launch the Amazon SageMaker training job.",
    )
```

**Reference (first 20 lines):**
```
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to print out the torch.distributed stack trace when something fails.",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the script to be launched in parallel, followed by all the arguments for the training "
            "script."
        ),
    )
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

#### Sample 33: argparse_real_034 (speedup: 0.62x)

- Source: pip/_internal/commands/freeze.py
- Accept: 0.29, Repair: 5, SQ: 0.794, OffStr: 0.000
- Failure reasons: A, B, D, F, H

**Prompt (first 20 lines):**
```
        self.cmd_opts.add_option(
            "-r",
            "--requirement",
            dest="requirements",
            action="append",
            default=[],
            metavar="file",
            help=(
                "Use the order in the given requirements file and its "
                "comments when generating output. This option can be "
                "used multiple times."
            ),
        )
        self.cmd_opts.add_option(
            "-l",
            "--local",
            dest="local",
            action="store_true",
            default=False,
            help=(
```

**Reference (first 20 lines):**
```
        self.cmd_opts.add_option(
            "--user",
            dest="user",
            action="store_true",
            default=False,
            help="Only output packages installed in user-site.",
        )
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.cmd_opts.add_option(
            "--all",
            dest="freeze_all",
            action="store_true",
            help=(
                "Do not skip these packages in the output:"
                " {}".format(", ".join(_dev_pkgs()))
            ),
        )
        self.cmd_opts.add_option(
            "--exclude-editable",
            dest="exclude_editable",
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

### Real-Python-DictConfig

#### Sample 59: dict_config_real_060 (speedup: 0.35x)

- Source: pip/_vendor/typing_extensions.py
- Accept: 0.17, Repair: 0, SQ: 0.714, OffStr: 0.000
- Failure reasons: A

**Prompt (first 20 lines):**
```
_PROTO_ALLOWLIST = {
    'collections.abc': [
        'Callable', 'Awaitable', 'Iterable', 'Iterator', 'AsyncIterable',
```

**Reference (first 20 lines):**
```
        'Hashable', 'Sized', 'Container', 'Collection', 'Reversible', 'Buffer',
    ],
    'contextlib': ['AbstractContextManager', 'AbstractAsyncContextManager'],
    'typing_extensions': ['Buffer'],
}
```

**Diagnosis**: Draft model cannot predict continuation tokens, leading to low acceptance.

#### Sample 2: dict_config_real_003 (speedup: 0.35x)

- Source: typing_extensions.py
- Accept: 0.17, Repair: 0, SQ: 0.714, OffStr: 0.000
- Failure reasons: A

**Prompt (first 20 lines):**
```
_PROTO_ALLOWLIST = {
    'collections.abc': [
        'Callable', 'Awaitable', 'Iterable', 'Iterator', 'AsyncIterable',
```

**Reference (first 20 lines):**
```
        'Hashable', 'Sized', 'Container', 'Collection', 'Reversible', 'Buffer',
    ],
    'contextlib': ['AbstractContextManager', 'AbstractAsyncContextManager'],
    'typing_extensions': ['Buffer'],
}
```

**Diagnosis**: Draft model cannot predict continuation tokens, leading to low acceptance.

#### Sample 1: dict_config_real_002 (speedup: 0.49x)

- Source: ipython_pygments_lexers.py
- Accept: 0.24, Repair: 0, SQ: 0.675, OffStr: 0.000
- Failure reasons: A, C, D

**Prompt (first 20 lines):**
```
    tokens = {
        "root": [
            # Tracebacks for syntax errors have a different style.
            # For both types of tracebacks, we mark the first line with
            # Generic.Traceback.  For syntax errors, we mark the filename
            # as we mark the filenames for non-syntax tracebacks.
```

**Reference (first 20 lines):**
```
            #
            # These two regexps define how IPythonConsoleLexer finds a
            # traceback.
            #
            ## Non-syntax traceback
            (r"^(\^C)?(-+\n)", bygroups(Error, Generic.Traceback)),
            ## Syntax traceback
            (
                r"^(  File)(.*)(, line )(\d+\n)",
                bygroups(
                    Generic.Traceback,
                    Name.Namespace,
                    Generic.Traceback,
                    Literal.Number.Integer,
                ),
            ),
            # (Exception Identifier)(Whitespace)(Traceback Message)
            (
                r"(?u)(^[^\d\W]\w*)(\s*)(Traceback.*?\n)",
                bygroups(Name.Exception, Generic.Whitespace, Text),
```

**Diagnosis**: Draft model cannot predict continuation tokens, leading to low acceptance.

### OpenMMLab-Config

#### Sample 70: openmmlab_config_real_071 (speedup: 0.50x)

- Source: _mmocr/configs/textdet/_base_/pretrain_runtime.py
- Accept: 0.24, Repair: 7, SQ: 0.685, OffStr: 0.118
- Failure reasons: A, B, G, H

**Prompt (first 20 lines):**
```
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1000),
    checkpoint=dict(
```

**Reference (first 20 lines):**
```
        type='CheckpointHook',
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=1),
)
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

#### Sample 64: openmmlab_config_real_065 (speedup: 0.62x)

- Source: _mmocr/configs/kie/_base_/default_runtime.py
- Accept: 0.28, Repair: 5, SQ: 0.722, OffStr: 0.000
- Failure reasons: A, B, H

**Prompt (first 20 lines):**
```
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
```

**Reference (first 20 lines):**
```
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False),
)
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

#### Sample 48: openmmlab_config_real_049 (speedup: 0.67x)

- Source: _mmpose/configs/_base_/default_runtime.py
- Accept: 0.34, Repair: 4, SQ: 0.820, OffStr: 0.100
- Failure reasons: A, B, G

**Prompt (first 20 lines):**
```
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
```

**Reference (first 20 lines):**
```
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5))
```

**Diagnosis**: Low draft-target agreement causes repeated repairs, creating overhead that negates TASD speedup.

### Rich-CLI-Option-Groups

#### Sample 21: rich_cli_option_groups_022 (speedup: 1.92x)

- Source: torch/_dynamo/repro/after_dynamo.py
- Accept: 1.00, Repair: 0, SQ: 0.542, OffStr: 1.000
- Failure reasons: F, G

**Prompt (first 20 lines):**
```
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="test accuracy",
        )
```

**Reference (first 20 lines):**
```
        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        parser.add_argument(
            "--no-isolate",
            dest="isolate",
            action="store_false",
            default=False,
            help="no isolate (doesn't do anything for after_dynamo)",
```

**Diagnosis**: Key ordering divergence between draft and target causes repeated rejections.

#### Sample 76: rich_cli_option_groups_077 (speedup: 1.93x)

- Source: pip/_internal/commands/freeze.py
- Accept: 1.00, Repair: 0, SQ: 0.550, OffStr: 1.000
- Failure reasons: D, F, G

**Prompt (first 20 lines):**
```
        self.cmd_opts.add_option(
            "-r",
            "--requirement",
            dest="requirements",
            action="append",
            default=[],
            metavar="file",
            help=(
                "Use the order in the given requirements file and its "
                "comments when generating output. This option can be "
                "used multiple times."
            ),
        )
        self.cmd_opts.add_option(
            "-l",
            "--local",
            dest="local",
            action="store_true",
            default=False,
            help=(
```

**Reference (first 20 lines):**
```
        self.cmd_opts.add_option(
            "--user",
            dest="user",
            action="store_true",
            default=False,
            help="Only output packages installed in user-site.",
        )
        self.cmd_opts.add_option(cmdoptions.list_path())
        self.cmd_opts.add_option(
            "--all",
            dest="freeze_all",
            action="store_true",
            help=(
                "Do not skip these packages in the output:"
                " {}".format(", ".join(_dev_pkgs()))
            ),
        )
        self.cmd_opts.add_option(
            "--exclude-editable",
            dest="exclude_editable",
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

#### Sample 38: rich_cli_option_groups_039 (speedup: 1.94x)

- Source: mistune/__main__.py
- Accept: 1.00, Repair: 0, SQ: 0.745, OffStr: 0.333
- Failure reasons: D, F, G

**Prompt (first 20 lines):**
```
    parser.add_argument(
        "-m",
        "--message",
        help="the markdown message to convert",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="the markdown file to convert",
    )
```

**Reference (first 20 lines):**
```
    parser.add_argument(
        "-p",
        "--plugin",
        metavar="NAME",
        action="extend",
        nargs="+",
        help="specifiy a plugin to use",
    )
    parser.add_argument(
        "--escape",
        action="store_true",
        help="turn on escape option",
    )
    parser.add_argument(
        "--hardwrap",
        action="store_true",
        help="turn on hardwrap option",
    )
    parser.add_argument(
        "-o",
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

### Complex-Nested-Config

#### Sample 1: complex_nested_config_002 (speedup: 1.93x)

- Source: modelscope/models/nlp/task_models/task_model.py
- Accept: 1.00, Repair: 0, SQ: 0.932, OffStr: 0.059
- Failure reasons: D, E, F, G

**Prompt (first 20 lines):**
```
                    local_metadata = {} if metadata is None else metadata.get(

                        prefix[:-1], {})

                    args = (state_dict, prefix, local_metadata, True, [], [],

                            error_msgs)

                    module._load_from_state_dict(*args)

                    for name, child in module._modules.items():

                        if child is not None:

                            load(child, prefix + name + '.')
```

**Reference (first 20 lines):**
```


                load(model_to_load, prefix=start_prefix)



            return error_msgs



        # Whole checkpoint

        mismatched_keys = _find_mismatched_keys(

            state_dict,

            model_state_dict,

            original_loaded_keys,

```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

#### Sample 9: complex_nested_config_010 (speedup: 1.93x)

- Source: debugpy/_vendored/pydevd/_pydevd_bundle/_debug_adapter/pydevd_schema.py
- Accept: 1.00, Repair: 0, SQ: 0.711, OffStr: 0.056
- Failure reasons: E, F, G, H

**Prompt (first 20 lines):**
```
    __props__ = {

        "seq": {

            "type": "integer",

            "description": "Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.",

        },

        "type": {"type": "string", "enum": ["response"]},

        "request_seq": {"type": "integer", "description": "Sequence number of the corresponding request."},

        "success": {
```

**Reference (first 20 lines):**
```
            "type": "boolean",

            "description": "Outcome of the request.\nIf True, the request was successful and the `body` attribute may contain the result of the request.\nIf the value is false, the attribute `message` contains the error in short form and the `body` may contain additional information (see `ErrorResponse.body.error`).",

        },

        "command": {"type": "string", "description": "The command requested."},

        "message": {

            "type": "string",

            "description": "Contains the raw error in short form if `success` is false.\nThis raw error might be interpreted by the client and is not shown in the UI.\nSome predefined values exist.",

            "_enum": ["cancelled", "notStopped"],

            "enumDescriptions": ["the request was cancelled.", "the request may be retried once the adapter is in a 'stopped' state."],

        },

```

**Diagnosis**: Key ordering divergence between draft and target causes repeated rejections.

#### Sample 69: complex_nested_config_070 (speedup: 1.94x)

- Source: debugpy/_vendored/pydevd/_pydevd_bundle/_debug_adapter/pydevd_schema.py
- Accept: 1.00, Repair: 0, SQ: 0.688, OffStr: 0.000
- Failure reasons: D, E

**Prompt (first 20 lines):**
```
    __props__ = {

        "seq": {

            "type": "integer",

            "description": "Sequence number of the message (also known as message ID). The `seq` for the first message sent by a client or debug adapter is 1, and for each subsequent message is 1 greater than the previous message sent by that actor. `seq` can be used to order requests, responses, and events, and to associate requests with their corresponding responses. For protocol messages of type `request` the sequence number can be used to cancel the request.",
```

**Reference (first 20 lines):**
```
        },

        "type": {"type": "string", "enum": ["event"]},

        "event": {"type": "string", "enum": ["capabilities"]},

        "body": {

            "type": "object",

            "properties": {"capabilities": {"$ref": "#/definitions/Capabilities", "description": "The set of updated capabilities."}},

            "required": ["capabilities"],

        },

    }
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

### Pipeline-Stage-Config

#### Sample 18: pipeline_stage_config_019 (speedup: 1.52x)

- Source: configs/fast_rcnn/fast-rcnn_r50_fpn_1x_coco.py
- Accept: 0.74, Repair: 1, SQ: 0.756, OffStr: 0.333
- Failure reasons: D, G, H

**Prompt (first 20 lines):**
```
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadProposals', num_max_proposals=2000),
    dict(type='LoadAnnotations', with_bbox=True),
```

**Reference (first 20 lines):**
```
    dict(
        type='ProposalBroadcaster',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackDetInputs')
]
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

#### Sample 66: pipeline_stage_config_067 (speedup: 1.85x)

- Source: _mmocr/configs/textdet/fcenet/fcenet_resnet50-dcnv2_fpn_1500e_ctw1500.py
- Accept: 1.00, Repair: 0, SQ: 0.773, OffStr: 0.400
- Failure reasons: D, G

**Prompt (first 20 lines):**
```
ctw_test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(type='Resize', scale=(1080, 736), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
```

**Reference (first 20 lines):**
```
    # does not need to do resize data transform
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

#### Sample 35: pipeline_stage_config_036 (speedup: 1.95x)

- Source: _mmpretrain/configs/simmim/benchmarks/swin-large-w14_8xb256-coslr-100e_in1k.py
- Accept: 1.00, Repair: 0, SQ: 0.850, OffStr: 0.100
- Failure reasons: D, G, H

**Prompt (first 20 lines):**
```
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
```

**Reference (first 20 lines):**
```
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=0.3333333333333333,
        fill_color=[103.53, 116.28, 123.675],
        fill_std=[57.375, 57.12, 58.395]),
    dict(type='PackInputs')
]
```

**Diagnosis**: Path/filename-specific strings are hard for the draft model to predict.

## 6. Hard vs Strong Case Comparison

- **Hard cases**: 79 total, 24 performance-driven, 55 quality-flagged

| Metric | Hard Cases | Strong Cases |
|--------|------------|--------------|
| Avg Prompt Chars | 313.443 | 276.327 |
| Avg Reference Chars | 1132.658 | 1244.719 |
| Avg Nesting Depth | 0.949 | 0.415 |
| Avg Accept Rate | 0.814 | 1.0 |
| Avg Repair Count | 1.177 | 0 |
| Avg Total Drafted | 207.62 | 128 |
| Avg Total Accepted | 125.949 | 128 |
| Avg SQ | 0.737 | 0.941 |
| Avg Off-Structure | 0.162 | 0.001 |
| Avg Truncation | 0.095 | 0.073 |
| Avg Speedup | 1.633 | 2.076 |

## 7. Conclusions

### Are TASD failures mainly from outliers?

Yes. Hard cases are 79/480 (16.5%), concentrated in a minority.

### Which benchmarks are most prone to hard cases?

- Complex-Nested-Config: 25.0%
- Real-Python-DictConfig: 18.8%
- OpenMMLab-Config: 18.8%
- Real-Python-Argparse: 13.8%
- Rich-CLI-Option-Groups: 12.5%
- Pipeline-Stage-Config: 10.0%

### Are hard cases mainly driven by draft-target divergence?

Among all hard cases, Reason A (low accept / draft-target divergence) appears in 25/79 (31.6%). Among performance hard cases specifically, Reason A appears in 23/24 (95.8%). Performance hard cases have avg accept rate 0.42 vs strong cases 1.00. Quality-flagged cases have accept rate 1.0 and are driven by SQ/off-structure metrics, not performance issues.

### Does structure guard prevent quality collapse?

Partially. 52/79 hard cases have SQ < 0.75. Average SQ of hard cases is 0.737 vs strong cases 0.941. The guard prevents complete structural collapse, but quality-flagged cases (mostly from Rich-CLI and Complex-Nested) show that the guard's SQ metric can be lower when the continuation structure differs from the prompt seed. These are not performance failures — they run at full speedup but the SQ metric flags structural differences.

### Should TASD-F be an optional extension?

Yes. TASD-F addresses the low-accept hard cases where draft-target divergence persists. Since hard cases are a minority (~16%), TASD-F should be optional, activated only when runtime signals indicate persistent low acceptance.

### Do main conclusions still hold?

Yes. TASD remains faster on average across all six benchmarks while preserving comparable structural quality. Hard cases define the boundary of TASD's applicability rather than invalidating the main trend.

## Paper-Ready Paragraph

Across 480 main-benchmark samples, TASD failures are concentrated in a minority of hard cases (79/480, 16.5%) characterized by low draft-target agreement (avg accept rate 0.81), repeated repairs (avg 1.2), and highly context-specific continuations such as paths, comments, or unstable key ordering. These cases define the boundary of TASD's applicability rather than invalidating the main trend: TASD remains faster on average across all six benchmarks while preserving comparable structural quality. The hard-case analysis motivates TASD-F as an optional runtime fallback for persistent low-acceptance regions.
