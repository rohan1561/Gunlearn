for seed in 1 2 3 4 5 6
do
	#CUDA_VISIBLE_DEVICES=1 python -u main_forget.py --save_dir ./saves/saves_c10_random/${seed} --mask ./masks/mask_c10/1model_SA_best.pth.tar --unlearn Gan --indexes_to_replace  --unlearn_lr 0.01 --unlearn_epochs 3  --dataset cifar10 --num_classes 10 --seed ${seed}
	CUDA_VISIBLE_DEVICES=3 python -u main_forget.py --save_dir ./saves/saves_c100_random/${seed} --mask ./masks/mask_c100/1model_SA_best.pth.tar --unlearn Gan --indexes_to_replace --unlearn_lr 0.01 --unlearn_epochs 5  --dataset cifar100 --num_classes 100 --seed ${seed}
done
