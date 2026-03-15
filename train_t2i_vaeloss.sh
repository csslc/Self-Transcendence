cd .../self-transcedence

accelerate launch train_t2i_vaeloss.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="exps" \
  --exp-name="t2i_vaeloss" \
  --data-dir='.../data/coco256_features'
