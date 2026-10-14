import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from protonets.models.utils import euclidean_dist
from protonets.utils import filter_opt
import protonets.utils.data as data_utils


METHODS = ('baseline', 'full', 'random', 'uncertainty')


def _mean_ci95(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0, 0.0
    std = float(values.std(ddof=1))
    half_width = 1.96 * std / math.sqrt(values.size)
    return mean, std, float(half_width)


def _targets(n_class, n_query, device):
    return torch.arange(n_class, device=device).view(n_class, 1).expand(
        n_class, n_query
    ).contiguous().view(-1)


def _prototypical_probabilities(model, xs, xq):
    n_class = xs.size(0)
    n_support = xs.size(1)
    n_query = xq.size(1)

    support = xs.view(n_class * n_support, *xs.size()[2:])
    query = xq.view(n_class * n_query, *xq.size()[2:])
    encoded = model.encoder(torch.cat([support, query], dim=0))
    z_dim = encoded.size(-1)
    prototypes = encoded[:n_class * n_support].view(
        n_class, n_support, z_dim
    ).mean(dim=1)
    query_embeddings = encoded[n_class * n_support:]
    probabilities = F.softmax(
        -euclidean_dist(query_embeddings, prototypes), dim=1
    )
    return probabilities, prototypes, query_embeddings


def _sample_affine_parameters(count, max_rotation, min_scale, max_scale,
                              max_translate, height, width, device, dtype,
                              generator):
    angles = (
        torch.rand(count, generator=generator, dtype=dtype) * 2.0 - 1.0
    ) * math.radians(max_rotation)
    scales = min_scale + torch.rand(
        count, generator=generator, dtype=dtype
    ) * (max_scale - min_scale)
    translate_x = torch.randint(
        -max_translate, max_translate + 1, (count,), generator=generator
    ).to(dtype=dtype)
    translate_y = torch.randint(
        -max_translate, max_translate + 1, (count,), generator=generator
    ).to(dtype=dtype)

    cos_values = torch.cos(angles) / scales
    sin_values = torch.sin(angles) / scales
    theta = torch.zeros(count, 2, 3, dtype=dtype)
    theta[:, 0, 0] = cos_values
    theta[:, 0, 1] = -sin_values
    theta[:, 1, 0] = sin_values
    theta[:, 1, 1] = cos_values
    theta[:, 0, 2] = 2.0 * translate_x / max(width - 1, 1)
    theta[:, 1, 2] = 2.0 * translate_y / max(height - 1, 1)
    return theta.to(device=device)


def _augment_query(query, config, generator):
    count, _, height, width = query.size()
    theta = _sample_affine_parameters(
        count=count,
        max_rotation=config['max_rotation'],
        min_scale=config['min_scale'],
        max_scale=config['max_scale'],
        max_translate=config['max_translate'],
        height=height,
        width=width,
        device=query.device,
        dtype=query.dtype,
        generator=generator
    )
    grid = F.affine_grid(theta, query.size(), align_corners=False)
    return F.grid_sample(
        query, grid, mode='bilinear', padding_mode='zeros',
        align_corners=False
    )


def _translate_query(query, shift_x, shift_y):
    translated = torch.zeros_like(query)
    height, width = query.size(-2), query.size(-1)
    src_x_start = max(0, -shift_x)
    src_x_end = width - max(0, shift_x)
    dst_x_start = max(0, shift_x)
    dst_x_end = width - max(0, -shift_x)
    src_y_start = max(0, -shift_y)
    src_y_end = height - max(0, shift_y)
    dst_y_start = max(0, shift_y)
    dst_y_end = height - max(0, -shift_y)
    translated[:, :, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = (
        query[:, :, src_y_start:src_y_end, src_x_start:src_x_end]
    )
    return translated


def _random_direction_views(query, views, generator):
    if not 1 <= views <= 4:
        raise ValueError('direction_subset requires between 1 and 4 views.')
    count = query.size(0)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    order = torch.rand(count, 4, generator=generator).argsort(dim=1)
    augmented_views = []
    for view_index in range(views):
        direction_ids = order[:, view_index].to(device=query.device)
        augmented = torch.zeros_like(query)
        for direction_id, (shift_x, shift_y) in enumerate(directions):
            mask = direction_ids.eq(direction_id)
            if mask.any():
                augmented[mask] = _translate_query(
                    query[mask], shift_x, shift_y
                )
        augmented_views.append(augmented)
    return augmented_views


def _augmented_embeddings(model, query, views, config, generator,
                          augmentation_mode):
    augmented_views = []
    if augmentation_mode == 'random_affine':
        for _ in range(views):
            augmented_views.append(_augment_query(query, config, generator))
    elif augmentation_mode == 'four_direction':
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
        if views != len(directions):
            raise ValueError('four_direction augmentation requires 4 views.')
        for shift_x, shift_y in directions:
            augmented_views.append(
                _translate_query(query, shift_x, shift_y)
            )
    elif augmentation_mode == 'direction_subset':
        augmented_views = _random_direction_views(
            query, views, generator
        )
    else:
        raise ValueError('Unknown augmentation mode: {}'.format(
            augmentation_mode
        ))
    return torch.stack([
        model.encoder(augmented) for augmented in augmented_views
    ], dim=0)


def _fuse_views(base_probabilities, base_embeddings,
                augmented_embeddings, prototypes, fusion_mode):
    if fusion_mode == 'probability_mean':
        augmented_probabilities = torch.stack([
            F.softmax(-euclidean_dist(embeddings, prototypes), dim=1)
            for embeddings in augmented_embeddings
        ], dim=0)
        return torch.cat([
            base_probabilities.unsqueeze(0), augmented_probabilities
        ], dim=0).mean(dim=0)
    if fusion_mode == 'embedding_mean':
        mean_embeddings = torch.cat([
            base_embeddings.unsqueeze(0), augmented_embeddings
        ], dim=0).mean(dim=0)
        return F.softmax(
            -euclidean_dist(mean_embeddings, prototypes), dim=1
        )
    raise ValueError('Unknown fusion mode: {}'.format(fusion_mode))


def _accuracy_and_loss(probabilities, targets):
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    loss = F.nll_loss(log_probabilities, targets)
    accuracy = probabilities.argmax(dim=1).eq(targets).float().mean()
    return float(loss.item()), float(accuracy.item())


def _selected_error_rate(base_probabilities, targets, selected):
    if selected.numel() == 0:
        return 0.0
    errors = base_probabilities.argmax(dim=1).ne(targets).float()
    return float(errors[selected].mean().item())


def _prediction_transitions(base_probabilities, new_probabilities, targets):
    base_correct = base_probabilities.argmax(dim=1).eq(targets)
    new_correct = new_probabilities.argmax(dim=1).eq(targets)
    return {
        'wrong_to_correct': int((~base_correct & new_correct).sum().item()),
        'correct_to_wrong': int((base_correct & ~new_correct).sum().item()),
        'wrong_to_wrong': int((~base_correct & ~new_correct).sum().item()),
        'correct_to_correct': int((base_correct & new_correct).sum().item())
    }


def _uncertainty_scores(probabilities, metric):
    if metric == 'entropy':
        n_class = probabilities.size(1)
        return -(probabilities.clamp_min(1e-12).log() *
                 probabilities).sum(dim=1) / math.log(n_class)
    if metric == 'max_probability':
        return 1.0 - probabilities.max(dim=1).values
    if metric == 'top2_margin':
        top_two = probabilities.topk(2, dim=1).values
        return 1.0 - (top_two[:, 0] - top_two[:, 1])
    raise ValueError('Unknown uncertainty metric: {}'.format(metric))


def _episode_metrics(model, sample, ratio, views, augmentation_config,
                     selection_generator, augmentation_generator,
                     augmentation_mode='random_affine',
                     fusion_mode='probability_mean',
                     selection_metric='entropy'):
    xs = sample['xs']
    xq = sample['xq']
    n_class = xs.size(0)
    n_query = xq.size(1)
    query = xq.view(n_class * n_query, *xq.size()[2:])
    targets = _targets(n_class, n_query, query.device)

    base_probabilities, prototypes, base_embeddings = _prototypical_probabilities(
        model, xs, xq
    )
    augmented_embeddings = _augmented_embeddings(
        model, query, views, augmentation_config, augmentation_generator,
        augmentation_mode
    )
    multi_view_probabilities = _fuse_views(
        base_probabilities, base_embeddings, augmented_embeddings,
        prototypes, fusion_mode
    )

    count = query.size(0)
    selected_count = min(count, max(1, int(math.ceil(count * ratio))))
    entropy = _uncertainty_scores(base_probabilities, 'entropy')
    uncertainty_scores = _uncertainty_scores(
        base_probabilities, selection_metric
    )
    uncertainty_indices = uncertainty_scores.topk(selected_count).indices
    random_indices = torch.randperm(
        count, generator=selection_generator
    )[:selected_count].to(device=query.device)

    probabilities = {
        'baseline': base_probabilities,
        'full': multi_view_probabilities,
        'random': base_probabilities.clone(),
        'uncertainty': base_probabilities.clone()
    }
    probabilities['random'][random_indices] = multi_view_probabilities[
        random_indices
    ]
    probabilities['uncertainty'][uncertainty_indices] = (
        multi_view_probabilities[uncertainty_indices]
    )

    metrics = {}
    for method in METHODS:
        loss, accuracy = _accuracy_and_loss(probabilities[method], targets)
        metrics[method] = {
            'loss': loss,
            'accuracy': accuracy,
            'transitions': _prediction_transitions(
                base_probabilities, probabilities[method], targets
            )
        }

    all_indices = torch.arange(count, device=query.device)
    uncertainty_mask = torch.ones(count, dtype=torch.bool, device=query.device)
    uncertainty_mask[uncertainty_indices] = False
    unselected_indices = all_indices[uncertainty_mask]

    metrics['diagnostics'] = {
        'selected_count': selected_count,
        'query_count': count,
        'mean_entropy': float(entropy.mean().item()),
        'mean_selection_score': float(uncertainty_scores.mean().item()),
        'uncertainty_selected_baseline_error_rate': _selected_error_rate(
            base_probabilities, targets, uncertainty_indices
        ),
        'uncertainty_unselected_baseline_error_rate': _selected_error_rate(
            base_probabilities, targets, unselected_indices
        )
    }
    return metrics


def _build_data_options(model_opt, opt):
    data_opt = {'data.' + key: value for key, value in
                filter_opt(model_opt, 'data').items()}
    data_opt['data.test_way'] = opt['data.way']
    data_opt['data.test_shot'] = opt['data.shot']
    data_opt['data.test_query'] = opt['data.query']
    data_opt['data.test_episodes'] = opt['data.episodes']
    return data_opt


def _summarize(records, ratio, views):
    summary = {
        'episodes': len(records),
        'selection_ratio': ratio,
        'extra_views': views,
        'methods': {},
        'paired_accuracy_difference': {},
        'diagnostics': {}
    }
    for method in METHODS:
        accuracies = [record[method]['accuracy'] for record in records]
        losses = [record[method]['loss'] for record in records]
        accuracy_mean, accuracy_std, accuracy_ci = _mean_ci95(accuracies)
        loss_mean, loss_std, loss_ci = _mean_ci95(losses)
        selected_ratio = 0.0 if method == 'baseline' else (
            1.0 if method == 'full' else ratio
        )
        summary['methods'][method] = {
            'accuracy_mean': accuracy_mean,
            'accuracy_std_across_episodes': accuracy_std,
            'accuracy_ci95_half_width': accuracy_ci,
            'loss_mean': loss_mean,
            'loss_std_across_episodes': loss_std,
            'loss_ci95_half_width': loss_ci,
            'theoretical_query_view_multiplier': 1.0 + selected_ratio * views,
            'prediction_transitions': {
                field: sum(record[method]['transitions'][field]
                           for record in records)
                for field in (
                    'wrong_to_correct', 'correct_to_wrong',
                    'wrong_to_wrong', 'correct_to_correct'
                )
            }
        }
        transitions = summary['methods'][method]['prediction_transitions']
        transitions['net_corrected'] = (
            transitions['wrong_to_correct'] - transitions['correct_to_wrong']
        )

    for method in ('full', 'random', 'uncertainty'):
        differences = [
            record[method]['accuracy'] - record['baseline']['accuracy']
            for record in records
        ]
        mean, std, ci = _mean_ci95(differences)
        summary['paired_accuracy_difference'][method + '_minus_baseline'] = {
            'mean': mean,
            'std_across_episodes': std,
            'ci95_half_width': ci
        }

    differences = [
        record['uncertainty']['accuracy'] - record['random']['accuracy']
        for record in records
    ]
    mean, std, ci = _mean_ci95(differences)
    summary['paired_accuracy_difference']['uncertainty_minus_random'] = {
        'mean': mean,
        'std_across_episodes': std,
        'ci95_half_width': ci
    }

    for field in (
        'mean_entropy',
        'mean_selection_score',
        'uncertainty_selected_baseline_error_rate',
        'uncertainty_unselected_baseline_error_rate'
    ):
        mean, std, ci = _mean_ci95([
            record['diagnostics'][field] for record in records
        ])
        summary['diagnostics'][field] = {
            'mean': mean,
            'std_across_episodes': std,
            'ci95_half_width': ci
        }
    return summary


def main(opt):
    output_dir = opt['eval.output_dir']
    os.makedirs(output_dir, exist_ok=True)

    random.seed(opt['eval.seed'])
    np.random.seed(opt['eval.seed'])
    torch.manual_seed(opt['eval.seed'])

    try:
        model = torch.load(
            opt['model.model_path'], map_location='cpu', weights_only=False
        )
    except TypeError:
        # PyTorch versions before the weights_only argument was introduced.
        model = torch.load(opt['model.model_path'], map_location='cpu')
    model.eval()

    model_opt_file = os.path.join(
        os.path.dirname(opt['model.model_path']), 'opt.json'
    )
    with open(model_opt_file, 'r') as handle:
        model_opt = json.load(handle)

    data_opt = _build_data_options(model_opt, opt)
    if data_opt['data.cuda']:
        if not torch.cuda.is_available():
            raise RuntimeError('The saved configuration requests CUDA, but CUDA is unavailable.')
        torch.cuda.manual_seed_all(opt['eval.seed'])
        model.cuda()

    split = opt['data.split']
    data = data_utils.load(data_opt, [split])
    selection_generator = torch.Generator().manual_seed(
        opt['eval.seed'] + 1
    )
    augmentation_generator = torch.Generator().manual_seed(
        opt['eval.seed'] + 2
    )
    augmentation_config = {
        'max_rotation': opt['augmentation.max_rotation'],
        'min_scale': opt['augmentation.min_scale'],
        'max_scale': opt['augmentation.max_scale'],
        'max_translate': opt['augmentation.max_translate']
    }
    augmentation_mode = opt['augmentation.mode']
    fusion_mode = opt['fusion.mode']
    selection_metric = opt['selection.metric']

    config = {
        'model_path': opt['model.model_path'],
        'split': split,
        'way': opt['data.way'],
        'shot': opt['data.shot'],
        'query_per_class': opt['data.query'],
        'episodes': opt['data.episodes'],
        'seed': opt['eval.seed'],
        'selection_ratio': opt['augmentation.ratio'],
        'extra_views': opt['augmentation.views'],
        'augmentation_mode': augmentation_mode,
        'fusion_mode': fusion_mode,
        'selection_metric': selection_metric,
        'augmentation': augmentation_config,
        'model_parameters_updated': False,
        'torch_version': torch.__version__
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write('\n')

    records = []
    start_time = time.time()
    with torch.no_grad():
        for episode_index, sample in enumerate(
                tqdm(data[split], desc='feasibility-' + split), start=1):
            metrics = _episode_metrics(
                model=model,
                sample=sample,
                ratio=opt['augmentation.ratio'],
                views=opt['augmentation.views'],
                augmentation_config=augmentation_config,
                selection_generator=selection_generator,
                augmentation_generator=augmentation_generator,
                augmentation_mode=augmentation_mode,
                fusion_mode=fusion_mode,
                selection_metric=selection_metric
            )
            metrics['episode'] = episode_index
            records.append(metrics)

    episode_path = os.path.join(output_dir, 'episode_metrics.jsonl')
    with open(episode_path, 'w') as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + '\n')

    summary = _summarize(
        records, opt['augmentation.ratio'], opt['augmentation.views']
    )
    summary['wall_time_seconds_for_joint_run'] = time.time() - start_time
    summary['timing_note'] = (
        'All augmented views are computed once and shared by the three '
        'augmentation methods. Use theoretical_query_view_multiplier for '
        'the controlled compute comparison.'
    )
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write('\n')

    print(json.dumps(summary, indent=2, sort_keys=True))
    print('Saved feasibility results to {}'.format(os.path.abspath(output_dir)))

