CUDA_VISIBLE_DEVICES=5 python main_vit.py &
CUDA_VISIBLE_DEVICES=4 python main_Uni.py &
CUDA_VISIBLE_DEVICES=3 python main_MPSA.py &
CUDA_VISIBLE_DEVICES=2 python main_Meta.py &
CUDA_VISIBLE_DEVICES=1 python main_cn_large.py &
wait