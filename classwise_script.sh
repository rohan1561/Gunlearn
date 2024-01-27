#for class in 0 1 2 3 4 5 6
for class in 0 #1 2 3 4 5 6
do
	#CUDA_VISIBLE_DEVICES=3 python -u main_forget.py --save_dir ./saves/saves_c10_classwise/${class} --mask ./masks/mask_c10/1model_SA_best.pth.tar --unlearn Gan --class_to_replace ${class} --num_indexes_to_replace 4500 --unlearn_lr 0.02 --unlearn_epochs 3  --dataset cifar10 --num_classes 10
	CUDA_VISIBLE_DEVICES=3 python -u main_forget.py --save_dir ./saves/saves_c100_classwise/${class} --mask ./masks/mask_c100/1model_SA_best.pth.tar --unlearn Gan --class_to_replace ${class} --num_indexes_to_replace 450 --unlearn_lr 0.01 --unlearn_epochs 5  --dataset cifar100 --num_classes 100 
done
