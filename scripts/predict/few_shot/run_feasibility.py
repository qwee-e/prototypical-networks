import argparse

from feasibility import main


parser = argparse.ArgumentParser(
    description='Evaluate selective augmentation on validation or test episodes'
)
parser.add_argument(
    '--model.model_path', type=str,
    default='results/baseline_5way_1shot/best_model.pt'
)
parser.add_argument('--data.split', choices=['val', 'test'], default='val')
parser.add_argument('--data.way', type=int, default=5)
parser.add_argument('--data.shot', type=int, default=1)
parser.add_argument('--data.query', type=int, default=15)
parser.add_argument('--data.episodes', type=int, default=1000)
parser.add_argument('--eval.seed', type=int, default=1234)
parser.add_argument(
    '--eval.output_dir', type=str,
    default='results/uncertainty_5way_1shot/feasibility'
)
parser.add_argument('--augmentation.ratio', type=float, default=0.30)
parser.add_argument('--augmentation.views', type=int, default=3)
parser.add_argument(
    '--augmentation.mode',
    choices=['random_affine', 'four_direction', 'direction_subset'],
    default='random_affine'
)
parser.add_argument(
    '--fusion.mode', choices=['probability_mean', 'embedding_mean'],
    default='probability_mean'
)
parser.add_argument(
    '--selection.metric',
    choices=['entropy', 'max_probability', 'top2_margin'],
    default='entropy'
)
parser.add_argument('--augmentation.max_rotation', type=float, default=0.0)
parser.add_argument('--augmentation.min_scale', type=float, default=1.0)
parser.add_argument('--augmentation.max_scale', type=float, default=1.0)
parser.add_argument('--augmentation.max_translate', type=int, default=1)

args = vars(parser.parse_args())

if not 0.0 < args['augmentation.ratio'] <= 1.0:
    parser.error('--augmentation.ratio must be in (0, 1].')
if args['augmentation.views'] < 1:
    parser.error('--augmentation.views must be at least 1.')
if (args['augmentation.mode'] == 'four_direction' and
        args['augmentation.views'] != 4):
    parser.error('--augmentation.mode four_direction requires 4 views.')
if args['augmentation.min_scale'] <= 0:
    parser.error('--augmentation.min_scale must be positive.')
if args['augmentation.max_scale'] < args['augmentation.min_scale']:
    parser.error('--augmentation.max_scale must be >= min_scale.')
main(args)
