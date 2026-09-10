#!/usr/bin/env python3
"""Run one or more inference experiments from a YAML manifest."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / 'configs'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'suite',
        choices=('generation', 'generation_mmm', 'mlm', 'pubchem', 'retrieval', 'retrieval_mmm'),
    )
    parser.add_argument('experiments', nargs='*', help='Experiment names to run.')
    parser.add_argument('--dataset', help='Run every experiment for this dataset.')
    parser.add_argument('--all', action='store_true', help='Run the complete suite.')
    parser.add_argument('--list', action='store_true', help='List matching experiments.')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without running.')
    parser.add_argument('--device', help='Override the device, e.g. cuda:0 or cpu.')
    parser.add_argument('--batch-size', type=int, help='Override the inference batch size.')
    parser.add_argument(
        '--employ-mmm', action='store_true',
        help='Use the shared-mask model for retrieval (multi-modality masking).',
    )
    parser.add_argument(
        '--deduplicate-branches', action='store_true',
        help='Deduplicate identical branches per SMILES during MLM evaluation.',
    )
    return parser.parse_args()


def load_suite(name):
    path = CONFIG_DIR / f'{name}.yaml'
    with path.open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get('experiments'), dict):
        raise ValueError(f'{path} must contain an experiments mapping')
    return config


def select_experiments(config, args):
    experiments = config['experiments']
    requested = set(args.experiments)
    if sum((bool(args.all), bool(args.dataset), bool(requested))) > 1:
        raise ValueError('Use only one of experiment names, --dataset, or --all')
    unknown = requested.difference(experiments)
    if unknown:
        raise ValueError(f'Unknown experiment(s): {", ".join(sorted(unknown))}')

    if args.all:
        selected = list(experiments)
    elif args.dataset:
        selected = [
            name for name, spec in experiments.items()
            if spec.get('dataset') == args.dataset
        ]
        if not selected:
            raise ValueError(f'No experiments found for dataset: {args.dataset}')
    elif requested:
        selected = [name for name in experiments if name in requested]
    elif args.list:
        selected = list(experiments)
    else:
        raise ValueError('Choose experiment names, --dataset, --all, or --list')
    return selected


def merge_options(defaults, experiment):
    options = dict(defaults)
    options.update(experiment.get('options', {}))
    return options


def build_command(entrypoint, options):
    command = [sys.executable, str(REPO_ROOT / entrypoint)]
    for option, value in options.items():
        if value is True:
            command.append(option)
        elif value is not False and value is not None:
            command.extend((option, str(value)))
    return command


def validate_paths(options):
    for option in (
        '--test-model-path', '--rank-model-path', '--pubchem-embeds',
        '--pubchem-dataframe', '--dataset-path',
    ):
        value = options.get(option)
        if value and not (REPO_ROOT / value).is_file():
            raise FileNotFoundError(f'{option} does not exist: {value}')
    if '--ds' in options:
        dataset = options['--ds']
        split = options.get('--split', 'test')
        data_dir = options.get('--data-dir', 'datasets/vibench')
        path = REPO_ROOT / data_dir / dataset / f'{dataset}_{split}.lmdb'
        if not path.is_file():
            raise FileNotFoundError(f'Dataset does not exist: {path}')


def main():
    args = parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    if args.employ_mmm and args.suite != 'retrieval':
        raise ValueError('--employ-mmm is only valid for the retrieval suite')
    if args.deduplicate_branches and args.suite != 'mlm':
        raise ValueError('--deduplicate-branches is only valid for the mlm suite')
    config_name = (
        'retrieval_mmm'
        if args.suite == 'retrieval' and args.employ_mmm
        else args.suite
    )
    config = load_suite(config_name)
    selected = select_experiments(config, args)

    if args.list:
        for name in selected:
            spec = config['experiments'][name]
            print(f'{name:<28} dataset={spec["dataset"]}')
        return

    total = len(selected)
    for index, name in enumerate(selected, start=1):
        spec = config['experiments'][name]
        options = merge_options(config.get('defaults', {}), spec)
        if args.device is not None:
            options['--device'] = args.device
        if args.batch_size is not None:
            options['--batch-size'] = args.batch_size
        if args.deduplicate_branches:
            options['--deduplicate-branches'] = True
        validate_paths(options)
        command = build_command(config['entrypoint'], options)
        print(f'\n[Experiment {index}/{total}] {name}', flush=True)
        print(f'$ {shlex.join(command)}', flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
