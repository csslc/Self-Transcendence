
cd /home/notebook/data/group/slc/self-transcedence

accelerate launch --main_process_port=29508 train_selftrans.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --stu-depth=6 \
  --tea-depth=8 \
  --cfg_guide=30.0 \
  --output-dir="exps/self_trans" \
  --ckpt_guided_model="exps/vaeloss/sit-b-256/0200000.pt" \
  --t-range 0.4 0.7 \
  --batch-size=256 \
  --exp-name="sit-b" \
  --data-dir=".../data"
