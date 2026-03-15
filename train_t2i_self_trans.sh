cd .../self-transcedence


accelerate launch train_t2i_self_trans.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --proj-coeff=0.5 \
  --stu-depth=12 \
  --tea-depth=16 \
  --cfg_guide=5.0 \
  --output-dir="exps" \
  --ckpt_guided_model="exps/t2i_vaeloss/checkpoints/0150000.pt" \
  --exp-name="t2i_self_trans" \
  --data-dir='.../data/coco256_features'