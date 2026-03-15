
cd .../self-transcedence

torchrun --nnodes=1 --nproc_per_node=8 --master-port=29510 generate.py \
  --model SiT-XL/2 \
  --ckpt pretrained_models/self-trans/sit_xl/2000000.pt \
  --sample-dir test_results/self-trans/sit-xl \
  --num-fid-samples 50000 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.9 \
  --resolution=256 \
  --guidance-high=0.65