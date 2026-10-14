# Prototypical Networks for Few-shot Learning

Code for the NIPS 2017 paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf).

If you use this code, please cite our paper:

```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
 ```

## Training a prototypical network

### Install dependencies

* This code has been tested on Ubuntu 16.04 with Python 3.6 and PyTorch 0.4.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install [torchnet](https://github.com/pytorch/tnt) by running `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`.

### Set up the Omniglot dataset

* Run `sh download_omniglot.sh`.

### Train the model

* Run `python scripts/train/few_shot/run_train.py`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXP_DIR` is your desired output directory.
  * If you are running on a GPU you can pass in the option `--data.cuda`.
  * Use `--train.seed SEED` to run an independent, reproducible training seed.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run evaluation as: `python scripts/predict/few_shot/run_eval.py --model.model_path results/trainval/best_model.pt`.

### Run the validation-only selective augmentation feasibility study

After training the frozen 5-way 1-shot baseline, compare the baseline, full
augmentation, random 30% augmentation, and uncertainty-guided 30%
augmentation on the same validation episodes without updating model
parameters:

```bash
python scripts/predict/few_shot/run_feasibility.py \
  --model.model_path results/baseline_5way_1shot/best_model.pt \
  --data.split val \
  --data.way 5 \
  --data.shot 1 \
  --data.query 15 \
  --data.episodes 1000 \
  --eval.seed 1234 \
  --augmentation.ratio 0.20 \
  --augmentation.mode four_direction \
  --augmentation.views 4 \
  --fusion.mode probability_mean \
  --selection.metric entropy
```

Results are written to `results/uncertainty_5way_1shot/feasibility/` as the
resolved configuration, one JSON record per episode, and an aggregate summary.
The feasibility entry point intentionally rejects the test split so that method
choices are made using validation data only.

The selected validation configuration uses four deterministic one-pixel
translations and augments the 20% highest-entropy queries. Validation ablations
showed that rotation and scaling substantially degraded this frozen model,
while selection ratios above 20% produced no meaningful additional accuracy.
