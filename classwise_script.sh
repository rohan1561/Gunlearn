for class in 0 1 2 3 4 5 6
do
	CUDA_VISIBLE_DEVICES=0 python -u main_forget.py --save_dir ./saves_c10_classwise/${class} --mask ./mask_c10/1model_SA_best.pth.tar --unlearn Gan --class_to_replace ${class} --num_indexes_to_replace 4500 --unlearn_lr 0.02 --unlearn_epochs 2  --dataset cifar10 --num_classes 10
done
