import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Plot accuracy and compute for the selection-ratio study'
)
parser.add_argument(
    '--result_root', type=Path,
    default=Path('results/uncertainty_5way_1shot')
)
parser.add_argument(
    '--output', type=Path,
    default=Path('results/uncertainty_5way_1shot/ratio_study.png')
)
args = parser.parse_args()

summary_paths = {
    10: args.result_root / 'ratio_10' / 'summary.json',
    20: args.result_root / 'ratio_20' / 'summary.json',
    30: args.result_root / 'directional_probability' / 'summary.json',
    50: args.result_root / 'ratio_50' / 'summary.json',
    100: args.result_root / 'ratio_100' / 'summary.json'
}

ratios = []
accuracy_gains = []
ci_half_widths = []
compute = []
for ratio, path in summary_paths.items():
    with path.open('r') as handle:
        summary = json.load(handle)
    result = summary['methods']['uncertainty']
    paired = summary['paired_accuracy_difference'][
        'uncertainty_minus_baseline'
    ]
    ratios.append(ratio)
    accuracy_gains.append(paired['mean'] * 100.0)
    ci_half_widths.append(paired['ci95_half_width'] * 100.0)
    compute.append(result['theoretical_query_view_multiplier'])

fig, accuracy_axis = plt.subplots(figsize=(8, 5))
accuracy_line = accuracy_axis.errorbar(
    ratios, accuracy_gains, yerr=ci_half_widths, marker='o', capsize=4,
    linewidth=2, label='Paired accuracy gain'
)
accuracy_axis.set_xlabel('Selected high-uncertainty queries (%)')
accuracy_axis.set_ylabel('Accuracy gain over baseline (percentage points)')
accuracy_axis.grid(alpha=0.3)

compute_axis = accuracy_axis.twinx()
compute_line, = compute_axis.plot(
    ratios, compute, marker='s', linestyle='--', color='tab:orange',
    linewidth=2, label='Query-view multiplier'
)
compute_axis.set_ylabel('Theoretical query-view multiplier')

accuracy_axis.legend(
    [accuracy_line.lines[0], compute_line],
    ['Paired accuracy gain', 'Query-view multiplier'],
    loc='best'
)
fig.suptitle('Accuracy and Compute by Uncertainty Selection Ratio')
fig.tight_layout()
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=300)
plt.close(fig)
print('Saved {}'.format(args.output.resolve()))
