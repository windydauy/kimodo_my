# Kimodo Distillation (16->8, 100->20)

## Goal
- Teacher: existing Kimodo G1 16-layer denoiser (pretrained checkpoint)
- Student: 8-layer denoiser
- Distillation target: student trained on reduced 20-step timestep grid while matching teacher behavior trained on full schedule

## Loss (7+7)
- Teacher branch (main): 7-term Kimodo loss between student `pred_x0` and teacher `pred_x0`
- GT branch (aux): 7-term Kimodo loss between student `pred_x0` and dataset GT `x0`
- Final:
  - `L = teacher_weight * L_teacher7 + gt_weight * L_gt7`
  - default: `teacher_weight=0.8`, `gt_weight=0.2`

## Files
- `kimodo/distillation/loss.py`: 7+7 weighted loss
- `kimodo/distillation/train.py`: distillation training loop
- `kimodo/distillation/configs/distill_g1_100_to_20.yaml`: default config
- `scripts/train_distill_g1_100_to_20.py`: python entrypoint
- `scripts/run_distill_g1_100_to_20.sh`: shell wrapper

## Run
```bash
export HF_HOME=./huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
bash scripts/run_distill_g1_100_to_20.sh
```

Optional resume:
```bash
RESUME_PATH=outputs/g1_distill_16to8_100to20/checkpoints/step_00002000.pt \
  bash scripts/run_distill_g1_100_to_20.sh
```

## Notes
- `distillation.student_steps=20` controls the timestep grid used during distillation.
- `distillation.teacher_steps=100` is metadata for experiment tracking; teacher itself remains full-capacity pretrained.
- Student warm start from teacher is enabled via same-name same-shape parameter copy.
