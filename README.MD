
This is the repository for the submission titled Discriminative Adversarial Unlearning.

# Abstract
We introduce a novel machine unlearning framework founded upon the established principles of the min-max optimization paradigm. We capitalize on the capabilities of strong Membership Inference Attacks (MIA) to facilitate the unlearning of specific samples from a trained model. We consider the scenario of two networks, the attacker $\mathbf{A}$ and the trained defender $\mathbf{D}$ pitted against each other in an adversarial objective, wherein the attacker aims at teasing out the information of the data to be unlearned in order to infer membership, and the defender unlearns to defend the network against the attack, whilst preserving its general performance. The algorithm can be trained end-to-end using backpropagation, following the well known iterative min-max approach in updating the attacker and the defender. We additionally incorporate a self-supervised objective effectively addressing the feature space discrepancies between the forget set and the validation set, enhancing unlearning performance. Our proposed algorithm closely approximates the ideal benchmark of retraining from scratch for both random sample forgetting and class-wise forgetting schemes on standard machine-unlearning datasets. Specifically, on the class unlearning scheme, the method demonstrates near-optimal performance and comprehensively overcomes known methods over the random sample forgetting scheme across all metrics and multiple network pruning strategies.
# Requirements
```
conda env create -f environment.yml
```

# Commands

### Adversarial Unlearning (Method presented in the submission)
#### Cifar-10 Random forgetting
```
python -u main_forget.py --save_dir ./saves/saves_c10_random/${seed} --mask ./masks/mask_c10/1model_SA_best.pth.tar --unlearn Gan --indexes_to_replace  --unlearn_lr 0.01 --unlearn_epochs 3  --dataset cifar10 --num_classes 10 --seed ${seed}
```

#### Cifar-10 Class-wise forgetting
```
python -u main_forget.py --save_dir ./saves/saves_c10_classwise/${class} --mask ./masks/mask_c10/1model_SA_best.pth.tar --unlearn Gan --class_to_replace ${class} --num_indexes_to_replace 4500 --unlearn_lr 0.02 --unlearn_epochs 3  --dataset cifar10 --num_classes 10 --stop-iter 30 --stop-unlearn-lr 0.01
```

#### Cifar-100 Random forgetting
```
python -u main_forget.py --save_dir ./saves/saves_c100_random/${seed} --mask ./masks/mask_c100/1model_SA_best.pth.tar --unlearn Gan --indexes_to_replace --unlearn_lr 0.01 --unlearn_epochs 5  --dataset cifar100 --num_classes 100 --seed ${seed}
```

#### Cifar-100 Class-wise forgetting
```
python -u main_forget.py --save_dir ./saves/saves_c100_classwise/${class} --mask ./masks/mask_c100/1model_SA_best.pth.tar --unlearn Gan --class_to_replace ${class} --num_indexes_to_replace 450 --unlearn_lr 0.02 --unlearn_epochs 5  --dataset cifar100 --num_classes 100 --stop-iter 30 --stop-unlearn-lr 0.02
```

### Acknowledgement
We acknowledge the valuable contributions of https://github.com/OPTML-Group/Unlearn-Sparse on whose work this repository is based.


