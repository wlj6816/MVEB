import math
import time
import shutil
import os 

import numpy as np
import torch
import torchvision

from .entropy  import entropy_gradeint
from .dataloader import build_dataloader
from .model import build_model
from .utils import AverageMeter
from .utils import concat_all_gather
from torch import nn, optim

def skip_nan_grad_backward(model, batch_idx):
    valid_gradients = True
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            if not valid_gradients:
                break

    if not valid_gradients:
        print(f"Deleted inf or nan values in gradients at batch_index {batch_idx}. Not updating model parameters.")
        model.zero_grad()


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])





class Pretrainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

        # build dataloader
        self.train_loader, self.memory_loader, self.test_loader = build_dataloader(self.cfg)
        self.total_steps = self.cfg.epochs * len(self.train_loader)
        self.warmup_steps = self.cfg.warmup_epochs * len(self.train_loader)
        
        # build model
        self.model = build_model(self.cfg)
        self.logger.info(f'{self.model}')

        # build optimizer
        self.optimizer = self.build_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()

        # build loss
        self.loss = getattr(self, self.cfg.loss)

    def build_optimizer(self):
        self.init_lr = self.cfg.base_lr * self.cfg.whole_batch_size / 256

        optim_params = self.model.module.parameters()
        param_weights = []
        param_biases = []
        for param in  self.model.module.parameters():
           if param.ndim == 1:
            param_biases.append(param)
           else:
            param_weights.append(param)
        optim_params = [{'params': param_weights}, {'params': param_biases}]

        optimizer = LARS(optim_params, lr=0, weight_decay=self.cfg.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)
        
        # optimizer = torch.optim.SGD(optim_params, self.init_lr,
        #                             momentum=self.cfg.momentum,
        #                             weight_decay=self.cfg.weight_decay)
        
        return optimizer
   
    def adjust_lr(self, optimizer, step):
        max_lr = self.init_lr
        min_lr = 1e-3 * self.init_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        
        optimizer.param_groups[0]['lr'] = lr 
        optimizer.param_groups[1]['lr'] = lr     
        
        return lr


    def adjust_mm(self, base_mm, step, schedule='cos'):
        if schedule == 'cos':
            return 1 - (1 - base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2
        elif schedule == 'const':
            return base_mm

    def resume(self, resume_path):
        ckpt = torch.load(resume_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scaler.load_state_dict(ckpt['scaler'])

        self.start_epoch = ckpt['epoch'] + 1
        self.step = ckpt['step']
        self.F = ckpt['F']

        if self.F is not None:
            self.F = self.F.cuda()

    def pretrain(self):
        self.last_saved_ckpt = None
        self.start_epoch, self.step = 0, 0
        self.F = None # EMA for correlation matrix

        # resume if required
        if self.cfg.resume_path is not None:
            self.resume(self.cfg.resume_path)

            # knn eval if required
            # be careful to do evaluation during training, the memory cost is too large
            knn_eval = getattr(self.cfg, 'knn_eval', False)
            if knn_eval:
                self.logger.info(f'{self.cfg.resume_path}')
                self.logger.info(f'knn: {self.knn_eval(self.model)}')
                return 0

        # begin training
        self.logger.info(f'Begin training, start_epoch:{self.start_epoch}, step:{self.step}')
        for epoch in range(self.start_epoch, self.cfg.epochs):
            if torch.distributed.is_available():
                self.train_loader.sampler.set_epoch(epoch)

            # collect epoch statistics
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses = AverageMeter('Loss', ':.4f')
            pos_sims = AverageMeter('Pos Sim', ':.4f')
           
            # switch to train mode
            self.model.train()
            
            end = time.time()
            # time.sleep(2)  # Prevent possible deadlock during epoch transition
            for i, data in enumerate(self.train_loader):
                # adjust lr and mm
                lr = self.adjust_lr(self.optimizer, self.step)
                mm = self.adjust_mm(self.cfg.base_momentum, self.step)
                self.step += 1

                with torch.cuda.amp.autocast():
                    images = data[0]
                    images = [im.cuda(non_blocking=True) for im in images]
                  
                    data_time.update(time.time() - end)
                    # forward
                    z_f, z1m, z2m = self.model(images, mm=mm)
                 
                    # print(time.time() - end)
         
                    # compute loss
                    loss_aliment,loss_unifo = self.loss(z_f, z1m, z2m)

                    pos_sim=loss_aliment.detach()
                    loss= -loss_aliment-self.cfg.lambd*loss_unifo
                    # exit if loss nan
                    if torch.any(torch.isnan(loss)):
                        print(f'{torch.cuda.current_device()} {loss}') 
                    return_flag = torch.tensor([0]).cuda()
                    if torch.isnan(loss):
                        return_flag = torch.tensor([1]).cuda()
                    torch.distributed.all_reduce(return_flag)
                    if return_flag:
                        self.logger.info(f"exit with loss value: {loss}")
                        return -1

                losses.update(loss.item(), images[0].size(0))
                pos_sims.update(pos_sim.item(), images[0].size(0))
                
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
               
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                skip_nan_grad_backward(self.model,i)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                if i==0 or (i+1) % self.cfg.print_freq == 0:
                    self.logger.info(f'Epoch: [{epoch}/{self.cfg.epochs}]  ' \
                                     f'iter: {i+1}/{len(self.train_loader)}  ' \
                                     f'{str(batch_time)}  ' \
                                     f'{str(data_time)}  ' \
                                     f'{str(losses)} ' \
                                     f'{str(pos_sims)} ' \
                                     f'lr: {lr} ' \
                                     f'mm: {mm}')

            # save model
            if torch.distributed.get_rank() == 0:
                if (epoch+1) % self.cfg.save_freq == 0:
                    ckpt_name = os.path.join(self.cfg.work_dir, 'checkpoint_{}.pth.tar'.format(epoch))
                else:
                    ckpt_name = os.path.join(self.cfg.work_dir, 'latest.pth.tar')
                self.logger.info('saving model')
                torch.save({'model': self.model.state_dict(),
                           'optimizer': self.optimizer.state_dict(),
                           'scaler': self.scaler.state_dict(),
                           'step': self.step,
                           'epoch': epoch,
                           'F': self.F,}, ckpt_name)
    
    def knn_eval(self, model):
        net = model.module.encoder
        projector = model.module.projector
        net.eval()
        projector.eval()
        classes = len(self.memory_loader.dataset.classes)
        total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            i = 0
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for data, target in self.memory_loader:
                feature = net(data.cuda(non_blocking=True))
                feature = torch.nn.functional.normalize(feature, dim=1)
                feature_bank.append(feature.clone())
                target_bank.append(target.cuda().clone())
                i += 1
                
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            target_bank = torch.cat(target_bank, dim=0).contiguous()
            
            tensors_gather = [torch.ones_like(feature_bank)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, feature_bank, async_op=False)

            feature_bank = torch.cat(tensors_gather, dim=-1)
            
            tensors_gather = [torch.ones_like(target_bank)
                for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, target_bank, async_op=False)

            target_bank = torch.cat(tensors_gather, dim=0)

            # loop test data to predict the label by weighted knn search
            i = 0
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            for data, target in self.test_loader:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data)
                feature = torch.nn.functional.normalize(feature, dim=1)

                pred_labels = self.knn_predict(feature, feature_bank, target_bank, classes, self.cfg.knn_k, self.cfg.knn_t)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                i += 1
            
            total_num = torch.tensor(total_num).cuda()
            total_top1 = torch.tensor(total_top1).cuda()
            torch.distributed.all_reduce(total_num)
            torch.distributed.all_reduce(total_top1)

        return total_top1.item() / total_num.item() * 100
    
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

    def loss_mveb(self,z,z1m, z2m):
        loss_aliment=0
        loss_uniform=0
        z_target=(z1m+ z2m)/2
        loss_aliment += torch.einsum('nc,nc->n', [z[0], z2m]).mean() 
        loss_aliment += torch.einsum('nc,nc->n', [z[1], z1m]).mean()
        loss_uniform += entropy_gradeint(z[0])
        loss_uniform += entropy_gradeint(z[1])
        
        for z_i in z[2:]:
            loss_aliment += torch.einsum('nc,nc->n', [z_i, z_target]).mean() 
            loss_uniform += entropy_gradeint(z_i)
        
        return loss_aliment/len(z),loss_uniform/len(z)

    