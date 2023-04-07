device=0
name=DualRefine_MR

CUDA_VISIBLE_DEVICES=$device \
python -m dualrefine.evaluate_depth \
    --load_weights_folder ./weights/$name \
    --eval_mono --batch_size 1
