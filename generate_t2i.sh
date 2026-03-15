
cd /home/notebook/data/group/slc/REPA




torchrun --nnodes=1 --nproc_per_node=8 --master-port=29533 generate_t2i.py \
  --ckpt exps/t2i_diffloss_newopt2_self_trans_cfg1/checkpoints/0150000.pt \
  --sample-dir test_results/t2i_10k/t2i_diffloss_newopt2_self_trans_cfg1 \
  --num-fid-samples 10000 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=2.0 \
  --resolution=256 \
  --guidance-high=1.0