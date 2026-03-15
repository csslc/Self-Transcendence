
cd .../self-transcedence


accelerate launch train_vaeloss.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --proj-coeff=0.5 \
  --encoder-depth=2 \
  --output-dir="exps/vaeloss" \
  --t-range 0.4 0.7 \
  --resolution 256 \
  --exp-name="sit-b-256" \
  --data-dir=".../data"