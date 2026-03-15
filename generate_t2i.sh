
cd /home/notebook/data/group/slc/REPA


torchrun --nnodes=1 --nproc_per_node=8 --master-port=29533 generate_t2i.py \
  --ckpt pretrained_models/self-trans/t2i/150000.pt \
  --sample-dir test_results/self-trans/t2i \
  --num-fid-samples 50000 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --resolution=256 \
  --guidance-high=1.0