#!/usr/bin/env python3
"""
Compare model metrics between two directories (e.g., raster-order vs random-order).
Outputs side-by-side comparison with differences highlighted.
"""

import json
import argparse
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

# Try to import rich for nice output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class MetricComparison:
    """Stores comparison result for a single metric."""
    name: str
    value_a: Optional[float] = None
    value_b: Optional[float] = None
    diff: Optional[float] = None
    diff_pct: Optional[float] = None
    is_better_b: Optional[bool] = None  # True if B is better (lower NLL/perplexity, higher accuracy)


def load_json_safe(filepath: Path) -> Optional[dict]:
    """Load JSON with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            # Fix malformed JSON with extra braces
            if content.count('{') != content.count('}'):
                depth = 0
                for i, char in enumerate(content):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            content = content[:i+1]
                            break
            return json.loads(content)
    except Exception as e:
        print(f"⚠️  Skip {filepath.name}: {e}")
        return None


def flatten_metrics(data: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict into dot-notation keys for easy comparison."""
    result = {}
    if not isinstance(data, dict):
        return result
    
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_metrics(value, full_key))
        elif isinstance(value, (int, float, str, bool)):
            result[full_key] = value
        # Skip lists for now (can be added if needed)
    return result


def get_better_direction(metric_name: str) -> str:
    """Return whether lower or higher is better for this metric."""
    lower_is_better = [
        'nll', 'perplexity', 'ece', 'overconfidence', 'gap', 'fid', 'loss', 'error'
    ]
    higher_is_better = [
        'accuracy', 'precision', 'recall', 'f1', 'confidence', 'auc'
    ]
    name_lower = metric_name.lower()
    if any(k in name_lower for k in lower_is_better):
        return 'lower'
    if any(k in name_lower for k in higher_is_better):
        return 'higher'
    return 'unknown'


def compare_values(name: str, val_a: Any, val_b: Any) -> MetricComparison:
    """Compare two numeric values and compute diff."""
    comp = MetricComparison(name=name)
    
    # Try to convert to float
    try:
        comp.value_a = float(val_a) if isinstance(val_a, (int, float)) else None
    except (ValueError, TypeError):
        comp.value_a = None
    
    try:
        comp.value_b = float(val_b) if isinstance(val_b, (int, float)) else None
    except (ValueError, TypeError):
        comp.value_b = None
    
    if comp.value_a is not None and comp.value_b is not None:
        comp.diff = comp.value_b - comp.value_a
        if comp.value_a != 0:
            comp.diff_pct = (comp.diff / abs(comp.value_a)) * 100
        
        direction = get_better_direction(name)
        if direction == 'lower':
            comp.is_better_b = comp.value_b < comp.value_a
        elif direction == 'higher':
            comp.is_better_b = comp.value_b > comp.value_a
    
    return comp


def find_matching_files(dir_a: Path, dir_b: Path, pattern: str = "*.json") -> list[tuple[str, Path, Path]]:
    """Find JSON files that exist in both directories."""
    files_a = {f.name: f for f in dir_a.rglob(pattern)}
    files_b = {f.name: f for f in dir_b.rglob(pattern)}
    
    # Filter to likely metrics files
    def is_metrics_file(name: str) -> bool:
        keywords = ['metric', 'eval', 'calibration', 'result', 'severity', 'random', 'raster']
        skip = ['cyclonedx', 'debugprotocol', 'pbr', 'package', 'tsconfig', 'launch', 'issue']
        name_lower = name.lower()
        if any(s in name_lower for s in skip):
            return False
        return any(k in name_lower for k in keywords)
    
    matches = []
    for name in sorted(set(files_a.keys()) & set(files_b.keys())):
        if is_metrics_file(name):
            matches.append((name, files_a[name], files_b[name]))
    
    return matches


def format_val(v: Optional[float], precision: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{precision}f}"


def print_comparison_table(
    comparisons: list[MetricComparison],
    label_a: str,
    label_b: str,
    use_rich: bool = True
) -> None:
    """Print side-by-side comparison table."""
    
    # Filter to numeric comparisons only
    numeric_comps = [c for c in comparisons if c.value_a is not None or c.value_b is not None]
    if not numeric_comps:
        print("⚠️  No numeric metrics found to compare")
        return
    
    if use_rich and RICH_AVAILABLE:
        _print_rich_table(numeric_comps, label_a, label_b)
    else:
        _print_plain_table(numeric_comps, label_a, label_b)


def _print_rich_table(comps: list[MetricComparison], label_a: str, label_b: str) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    table = Table(title="Metrics Comparison", box=box.ROUNDED, show_lines=True)
    
    table.add_column("Metric", style="cyan", width=40)
    table.add_column(label_a, justify="right", width=12)
    table.add_column(label_b, justify="right", width=12)
    table.add_column("Δ (B-A)", justify="right", width=10)
    table.add_column("Δ%", justify="right", width=8)
    # Removed "Better" column
    
    for c in comps:
        # Color-code the difference
        diff_text = Text(format_val(c.diff))
        if c.diff is not None:
            if c.is_better_b is True:
                diff_text.stylize("green")
            elif c.is_better_b is False:
                diff_text.stylize("red")
            elif c.diff > 0:
                diff_text.stylize("yellow")
            else:
                diff_text.stylize("blue")
        
        pct_str = f"{c.diff_pct:+.2f}%" if c.diff_pct is not None else "N/A"
        
        metric_name = c.name.split('.')[-1]  # Show only last part of key
        table.add_row(
            metric_name,
            format_val(c.value_a),
            format_val(c.value_b),
            diff_text.plain,
            pct_str
        )
    
    console.print(table)


def _print_plain_table(comps: list[MetricComparison], label_a: str, label_b: str) -> None:
    print(f"\n{'Metric':<40} {label_a:>12} {label_b:>12} {'Δ(B-A)':>10} {'Δ%':>8}")
    print("-" * 85)
    
    for c in comps:
        metric_name = c.name.split('.')[-1]
        diff_str = f"{c.diff:+.4f}" if c.diff is not None else "N/A"
        pct_str = f"{c.diff_pct:+.2f}%" if c.diff_pct is not None else "N/A"
        
        print(f"{metric_name:<40} {format_val(c.value_a):>12} {format_val(c.value_b):>12} {diff_str:>10} {pct_str:>8}")


def compare_files(
    file_a: Path,
    file_b: Path,
    label_a: str = "A",
    label_b: str = "B",
    use_rich: bool = True
) -> None:
    """Compare two JSON files and print differences."""
    print(f"\n📊 Comparing: {file_a.name}")
    print(f"   {label_a}: {file_a}")
    print(f"   {label_b}: {file_b}")
    print("=" * 80)
    
    data_a = load_json_safe(file_a)
    data_b = load_json_safe(file_b)
    
    if data_a is None or data_b is None:
        print("⚠️  Could not load one or both files")
        return
    
    # Flatten for easy comparison
    flat_a = flatten_metrics(data_a)
    flat_b = flatten_metrics(data_b)
    
    # Get all keys from both
    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    
    # Compare each metric
    comparisons = []
    for key in all_keys:
        val_a = flat_a.get(key)
        val_b = flat_b.get(key)
        comp = compare_values(key, val_a, val_b)
        if comp.value_a is not None or comp.value_b is not None:
            comparisons.append(comp)
    
    # Print results
    print_comparison_table(comparisons, label_a, label_b, use_rich)


def main():
    parser = argparse.ArgumentParser(
        description="Compare metrics between two directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s raster-order/ random-order/
  %(prog)s results/raster results/random --labels "Raster" "Random"
  %(prog)s . --file-a metrics_a.json --file-b metrics_b.json
        """
    )
    
    parser.add_argument('dir_a', nargs='?', help='First directory (e.g., raster-order)')
    parser.add_argument('dir_b', nargs='?', help='Second directory (e.g., random-order)')
    
    parser.add_argument('--labels', '-l', nargs=2, default=['A', 'B'],
                       help='Labels for the two directories (default: A B)')
    
    parser.add_argument('--pattern', '-p', default='*.json',
                       help='Glob pattern for metric files')
    
    parser.add_argument('--file-a', type=str, help='Compare single files instead of directories')
    parser.add_argument('--file-b', type=str, help='Second file for single-file comparison')
    
    parser.add_argument('--no-rich', action='store_true', help='Disable rich formatting')
    
    parser.add_argument('--key', '-k', type=str, help='Filter to metrics containing this key')
    
    args = parser.parse_args()
    
    use_rich = not args.no_rich and RICH_AVAILABLE
    
    # Single file mode
    if args.file_a and args.file_b:
        compare_files(
            Path(args.file_a), Path(args.file_b),
            args.labels[0], args.labels[1], use_rich
        )
        return
    
    # Directory mode
    if not args.dir_a or not args.dir_b:
        parser.print_help()
        return
    
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    
    if not dir_a.exists() or not dir_b.exists():
        print(f"❌ One or both directories not found")
        return
    
    # Find matching files
    matches = find_matching_files(dir_a, dir_b, args.pattern)
    
    if not matches:
        print(f"⚠️  No matching metric files found in {dir_a} and {dir_b}")
        return
    
    print(f"🔍 Found {len(matches)} matching file(s) to compare")
    
    for fname, path_a, path_b in matches:
        if args.key and args.key.lower() not in fname.lower():
            continue
        compare_files(path_a, path_b, args.labels[0], args.labels[1], use_rich)
        print()


if __name__ == "__main__":
    main()