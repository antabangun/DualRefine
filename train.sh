device=0
name=reproduce_m

CUDA_VISIBLE_DEVICES=$device \
python -m dualrefine.train \
    --log_dir weights --model_name $name \
    --mixed_precision