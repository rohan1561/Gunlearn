import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []


def Gan(dataloaders, modelD, modelA, args):
    iters = 0
    print("Starting Training Loop...")
    # For each epoch
    netA = modelA # Attacker
    netD = modelD # Defender
    netA.train()
    netD.train()
    num_epochs = args.unlearn_epochs
    forget_label = 1.
    test_label = 0.
    beta1 = 0.5
    a_iters = 1

    # Setup Adam optimizers for both G and D
    optimizerA = torch.optim.Adam(
        netA.parameters(),
        lr=0.001,
        #momentum=args.momentum,
        #weight_decay=args.weight_decay,
    )
    '''
    optimizerA = torch.optim.SGD(
        netA.parameters(),
        lr=0.01,
        #momentum=args.momentum,
        #weight_decay=args.weight_decay,
    )
    '''
    optimizerD = torch.optim.SGD(
        netD.parameters(),
        args.unlearn_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    # Setup dataloaders
    retain_loader = dataloaders['retain']
    test_prime_loader = dataloaders['val']
    forget_loader = dataloaders['train']
    m = 10
    '''
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )  # 0.1 is fixed
    '''
    if not args.train_attacker:
        forget_loader = dataloaders['forget']
        netA.load_state_dict(torch.load(os.path.join('attackers',\
                #f'attacker_pretrained_{args.dataset}_{args.seed}checkpoint.pth.tar'))['state_dict'])
                f'attacker_pretrained_{args.dataset}checkpoint.pth.tar'))['state_dict'])
        print(f'ATTACKER LOADED')
 
    for epoch in range(num_epochs):
        # For each batch in the dataloader

        for i, (data, labels) in enumerate(retain_loader, 0):
            ###########################
            # Update ATTACKER Model
            ###########################
            device = (
                torch.device("cuda:0") if torch.cuda.is_available()\
                        else torch.device("cpu"))
            '''
            if (epoch in [3, 7]) and (i == 0):
                a_iters = 1500
            else:
                a_iters = 1
            '''
            for j in range(a_iters):
                '''
                SCHEDULE THE LR FOR THE ATTACKER FOR BETTER PERFORMANCE!!
                '''
                fseta = next(iter(forget_loader))
                fset = fseta[0][:min(args.batch_size, len(test_prime_loader.dataset))].to(device)
                flabels = fseta[1][:min(args.batch_size, len(test_prime_loader.dataset))].to(device)
                f_gt = nn.functional.one_hot(flabels, num_classes=args.num_classes)
                f_size = fset.shape[0]
                fset_noisy = fset.repeat(m, 1, 1, 1) 
                uf = torch.randn_like(fset_noisy)
                fset_noisy = fset_noisy + 1e-3*uf

                tprimea = next(iter(test_prime_loader))
                tprime = tprimea[0].to(device)
                tlabels = tprimea[1].to(device)
                t_gt = nn.functional.one_hot(tlabels, num_classes=args.num_classes)
                t_size = tprime.shape[0]
                tprime_noisy = tprime.repeat(m, 1, 1, 1) 
                ut = torch.randn_like(tprime_noisy)
                tprime_noisy = tprime_noisy + 1e-3*ut

                flabel = torch.full((f_size,), forget_label, dtype=torch.long, device=device)
                tlabel = torch.full((t_size,), test_label, dtype=torch.long, device=device)
                assert flabel.shape == tlabel.shape
         
                dfpreds_fset, dffeats_fset = netD(fset)
                dffeats_fset = (dffeats_fset - dffeats_fset.mean(dim=0))/dffeats_fset.std(0)
                dfpreds_noisy, _ = netD(fset_noisy)
                difff = torch.abs(dfpreds_noisy - dfpreds_fset.repeat((m, 1)))
                difff = difff.view(m, -1, args.num_classes)
                difff = difff.mean(dim=0) / 1e-3
                dfpreds = torch.cat([dfpreds_fset.detach(), f_gt, difff.detach()], dim=1)

                dtpreds_tset, dtfeats_tset = netD(tprime)
                dtfeats_tset = (dtfeats_tset - dtfeats_tset.mean(dim=0))/dtfeats_tset.std(0)
                dtpreds_noisy, _ = netD(tprime_noisy)
                difft = torch.abs(dtpreds_noisy - dtpreds_tset.repeat((m, 1)))
                difft = difft.view(m, -1, args.num_classes)
                difft = difft.mean(dim=0) / 1e-3
 
                dtpreds = torch.cat([dtpreds_tset.detach(), t_gt, difft.detach()], dim=1)
                dpreds = torch.cat([dfpreds, dtpreds], dim=0)
                Apreds = netA(dpreds)
                lossA = criterion(Apreds, torch.cat([flabel, tlabel]))
                optimizerA.zero_grad()
                lossA.backward()
                optimizerA.step()

                clsacc = utils.accuracy(Apreds.data, torch.cat([flabel, tlabel]))[0]
                Defacc_f = utils.accuracy(dfpreds_fset.data, flabels.to(device))[0]
                Defacc_v = utils.accuracy(dtpreds_tset.data, tlabels)[0]
                print(clsacc.item(), Defacc_f.item(), Defacc_v.item(), iters,\
                        'attacker, def_forget, def_valid, iterations')
                iters += 1
                if (iters == 20000) & args.train_attacker:
                    print(iters, args.train_attacker, 'xxxxxxxxxxxxxxx')
                    state = {'state_dict': netA.state_dict(), 'acc_attack': clsacc}
                    utils.save_checkpoint(state, False, args.save_dir,\
                            f'attacker_pretrained_{args.dataset}_{args.seed}')
                    return
            if not args.train_attacker:
                ############################
                # Update DEFENDER network
                ###########################
                clpreds, _ = netD(data.to(device))
                errcl = criterion(clpreds, labels.to(device))
                dfpreds = torch.cat([dfpreds_fset, f_gt, difff], dim=1)
                dtpreds = torch.cat([dtpreds_tset, t_gt, difft], dim=1)
                dpreds = torch.cat([dfpreds, dtpreds], dim=0)
                Apreds = netA(dpreds)
                errA = criterion(Apreds, torch.cat([tlabel, flabel])) # REVERSE THE LABELS
                #errcl_f = criterion(dfpreds_fset, flabels)
                # contrastive objective
                '''
                # DINO
                dtprobs_tset = torch.mean(F.softmax(dtfeats_tset, dim=-1), dim=0)
                dfprobs_fset = torch.mean(F.log_softmax(dffeats_fset, dim=-1), dim=0)
                #print(dfprobs_fset.shape, dtprobs_tset.shape)
                contral = torch.sum(-dtprobs_tset.detach()*dfprobs_fset)
                contral = 0
                for lb in range(args.num_classes):
                    lmask_test = [tlabels == lb]
                    dtfeats_lb = dtfeats_tset[lmask_test]
                    lmask_forg = [flabels == lb]
                    dffeats_lb = dffeats_fset[lmask_forg]

                    dtfeats_mean = torch.mean(dtfeats_lb, dim=0)
                    dffeats_mean = torch.mean(dffeats_lb, dim=0)
                    contram = torch.outer(dtfeats_mean, dffeats_mean)
                    mask = torch.eye(contram.shape[0]).to(device)
                    posvals = torch.square(contram.masked_select(mask.bool()) - 1).sum()
                    negvals = torch.square(contram.masked_select(~mask.bool())).sum()
                    print(posvals, negvals, 'xxxxxxxxxxxxxx')
                    contral += posvals + 5e-4*negvals
                print(contral, 'xxxxxxxxxxxxxxxxxxxxxxxxx')
                '''
                # Barlow Twins
                #dtfeats_mean = torch.sum(dtfeats_tset, dim=0)
                #dffeats_mean = torch.sum(dffeats_fset, dim=0)
                #contram = torch.outer(dtfeats_mean, dffeats_mean)
                contram = torch.mean(torch.bmm(dtfeats_tset.unsqueeze(2),\
                        dffeats_fset.unsqueeze(1)), dim=0)
                mask = torch.eye(contram.shape[0]).to(device)
                posvals = torch.square(contram.masked_select(mask.bool()) - 1).sum()
                negvals = torch.square(contram.masked_select(~mask.bool())).sum()
                print(posvals, negvals, 'xxxxxxxxxxxxxx')
                contral = posvals + 5e-3*negvals
                beta = 1
                if args.class_to_replace in list(range(args.num_classes)):
                    if iters > 30:
                        beta = 0
                        for g in optimizerD.param_groups:
                            g['lr'] = 0.01

                lossD = errcl + beta*(0.001*contral +0.9*errA)  #- 0.004*errcl_f 
                print(errcl.item(), errA.item(), args.seed,\
                        'retaining set error, attacker error, DEFENDER UPDATE')
                optimizerD.zero_grad()
                lossD.backward()
                optimizerD.step()

                retacc = utils.accuracy(clpreds.data, labels.to(device))[0]
                print(retacc.item(), 'retaining set accuracy')
 
            # Output training stats
            if i % 1 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_A: %.4f'
                      % (epoch, num_epochs, i, len(retain_loader),
                         lossA.item(), lossA.item()))

