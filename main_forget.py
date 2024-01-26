import copy
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import pruner # extract mask, prune model, check sparsity
import unlearn # unlearning method and saving
import utils
from trainer import validate


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    print(args)
    (
        models,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    modelD, modelA = models
    modelD.cuda()
    modelA.cuda() 
    '''
    msg = modelD.load_state_dict(torch.load('./pretrained/resnet18c10.pt'), strict=False)
    print(msg)
    print('model loaded with the message above')
    #sys.exit()
    '''

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        print(len(forget_dataset))
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        print(len(retain_dataset))
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset), 'Forget Dataset length')
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            print('yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy')
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            print(len(forget_dataset))
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print(len(retain_dataset))
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    if args.class_to_replace is None:
        utils.dataset_convert_to_test(forget_loader.dataset, args)
        utils.dataset_convert_to_test(val_loader.dataset, args)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print(set(forget_loader.dataset.targets),\
            set(val_loader.dataset.targets))
    print(set(retain_loader.dataset.targets),\
            set(test_loader.dataset.targets))
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    unlearn_data_loaders = OrderedDict(train=train_loader_full,
        retain=retain_loader, forget=forget_loader, val=val_loader,\
                test=test_loader)

    ########### USING PRUNED MODEL FOR NOW. COMMENT OUT LATER###########
    checkpoint = torch.load(args.mask, map_location=device)
    if "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]
    current_mask = pruner.extract_mask(checkpoint)
    pruner.prune_model_custom(modelD, current_mask)
    pruner.check_sparsity(modelD)
    modelD.load_state_dict(checkpoint, strict=False)
    ########### USING PRUNED MODEL FOR NOW. COMMENT OUT LATER###########

    evaluation_result = None

    unlearn_method = unlearn.get_unlearn_method(args.unlearn)
    criterion = nn.CrossEntropyLoss()
    if args.unlearn == "Gan":
        unlearn_method(unlearn_data_loaders, modelD, modelA, args)
        if args.train_attacker:
            state = {'state_dict': modelA.state_dict()}
            utils.save_checkpoint(state, False, args.save_dir, f'attacker_pretrained_{args.dataset}')
            sys.exit()
    else:
        unlearn_method(unlearn_data_loaders, modelD, criterion, args)

    #unlearn.save_unlearn_checkpoint(model, None, args)
    model = modelD

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    """training privacy MIA:
        in distribution: retain
        out of distribution: test
        target: (retain, test)"""
    if "SVC_MIA_training_privacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        retain_len = len(retain_dataset)
        num = test_len // 2

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(num)))
        target_train = torch.utils.data.Subset(
            retain_dataset, list(range(num, retain_len))
        )
        shadow_test = torch.utils.data.Subset(test_loader.dataset, list(range(num)))
        target_test = torch.utils.data.Subset(
            test_loader.dataset, list(range(num, test_len))
        )

        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )
        shadow_test_loader = torch.utils.data.DataLoader(
            shadow_test, batch_size=args.batch_size, shuffle=False
        )

        target_train_loader = torch.utils.data.DataLoader(
            target_train, batch_size=args.batch_size, shuffle=False
        )
        target_test_loader = torch.utils.data.DataLoader(
            target_test, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_training_privacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=shadow_test_loader,
            target_train=target_train_loader,
            target_test=target_test_loader,
            model=model,
        )
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
