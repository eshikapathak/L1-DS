
# Training
python -m src.train.train_node --config configs/CShape/node_train.yaml

# (Optional) Override a few things from CLI
python -m src.train.train_node \
  --config configs/CShape/node_train.yaml \
  --steps 8000 --base_lr 3e-4 --width 128


# Run experiments on the newly trained shape
Uses the YAML to locate the model automatically, overlays all controllers, uses avg demo for DTW, and plots the disturbance subplot.
python -m src.experiments.run_experiments \
  --train_yaml configs/CShape/node_train.yaml \
  --out outputs/experiments/CShape_suite \
  --verbose

# LLC on, matched sine disturbance
python -m src.experiments.run_experiments \
  --train_yaml configs/CShape/node_train.yaml \
  --out outputs/experiments/CShape_LLC_matched_sine \
  --with_llc --matched --matched_type sine \
  --verbose

B) How to run (examples)
1) LLC on, no lower-level disturbance
python -m src.experiments.run_experiments \
  --train_yaml configs/Worm/node_train.yaml \
  --out outputs/experiments/Worm_LLC_none \
  --with_llc \
  --verbose

2) LLC on, matched sine disturbance (acts on acceleration channel)
python -m src.experiments.run_experiments \
  --train_yaml configs/Worm/node_train.yaml \
  --out outputs/experiments/Worm_LLC_matched_sine \
  --with_llc --matched --matched_type sine \
  --verbose

3) LLC on, unmatched pulse disturbance (acts on position-rate channel)
python -m src.experiments.run_experiments \
  --train_yaml configs/Worm/node_train.yaml \
  --out outputs/experiments/Worm_LLC_unmatched_pulse \
  --with_llc --unmatched --unmatched_type pulse \
  --verbose

4) LLC on, both matched & unmatched (mixed types if you want)
python -m src.experiments.run_experiments \
  --train_yaml configs/Worm/node_train.yaml \
  --out outputs/experiments/Worm_LLC_both \
  --with_llc \
  --matched --matched_type sine \
  --unmatched --unmatched_type pulse \
  --verbose
