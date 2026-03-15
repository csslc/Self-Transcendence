cd .../self-transcedence

# accelerate launch --main_process_port=29501 train_t2i_self_trans.py \
#   --report-to="wandb" \
#   --allow-tf32 \
#   --mixed-precision="fp16" \
#   --seed=0 \
#   --path-type="linear" \
#   --prediction="v" \
#   --weighting="uniform" \
#   --enc-type="dinov2-vit-b" \
#   --proj-coff=0.5 \
#   --encoder_depth_1=12 \
#   --encoder_depth_2=16 \
#   --cfg_guide=5.0 \
#   --output-dir="exps" \
#   --ckpt_guided_model="exps/t2i_diffloss_newopt2_trange/checkpoints/0150000.pt" \
#   --exp-name="t2i_diffloss_newopt2_self_trans_cfg5_new" \
#   --resume-step=250000 \
#   --data-dir='/home/notebook/data/group/slc/REPA/data/coco256_features'


accelerate launch --main_process_port=29501 train_t2i_self_trans.py \
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