import os

import torch
from torch import nn
from nets.deeplabv3_training import (CE_Loss, Dice_loss, Focal_Loss,CE_Cls_Loss,CE_loss_cls,
                                     weights_init)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score
import torch.nn.functional as F



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0,file_train=None,file_val=None,weight_fit=True,all_weight_fit=False):
    total_loss      = 0
    train_cls_loss  = 0
    train_seg_loss  = 0
    train_acc       = 0
    val_cls_loss    = 0

    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0
    val_accuracy    = 0
    val_seg_loss    = 0
    train_file = r'VOCdevkit_LINES_LAST/VOC2007/ImageSets/Segmentation/train.txt'
    val_file = r'VOCdevkit_LINES_LAST/VOC2007/ImageSets/Segmentation/val.txt'
    test_file = r'VOCdevkit_LINES_LAST/VOC2007/ImageSets/Segmentation/test.txt'
    #file_train = open(file=os.path.join(save_dir,'test_log.txt'),mode = 'w')
    #file_val = open(file=os.path.join(save_dir,'val_log.txt'),mode = 'w')
    focalLoss = FocalLoss()


    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break

        imgs, pngs, labels, cls_label = batch
        #imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                cls_label = cls_label.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            outputs,outputs_cls = model_train(imgs)

            if focal_loss:
                #print('focal')
                loss_seg = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss_cls = focalLoss(outputs_cls, cls_label)
                #print(loss_cls)
            else:
                cls_label=cls_label.to(torch.long)
                loss_cls = nn.CrossEntropyLoss()(outputs_cls, cls_label)

                loss_seg = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
            if weight_fit == True:
                blance = ((epoch + 1)/Epoch)
                loss = (1 - blance) * 0.7 * 0.5 * loss_seg + 0.5 * loss_cls
                
            if all_weight_fit == True:
                loss = (1 - blance) * 0.7 * 0.5 * loss_seg + blance * 0.5 * loss_cls
            if weight_fit==False and all_weight_fit==False:
                loss = 0.7 * 0.5 * loss_seg + 0.5 * loss_cls

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs,outputs_cls = model_train(imgs)
                
                if focal_loss:
                    loss_seg = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                    loss_cls = focalLoss(outputs_cls, cls_label)
                else:
        
                    cls_label=cls_label.to(torch.long)
                    loss_cls = nn.CrossEntropyLoss()(outputs_cls, cls_label)
                    loss_seg = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
                if weight_fit == True:
                    blance = ((epoch + 1)/Epoch)
                    loss = (1 - blance) * 0.7 * 0.5 * loss_seg + 0.5 * loss_cls

                if all_weight_fit == True:
                    loss = (1 - blance) * 0.7 * 0.5 * loss_seg + blance * 0.5 * loss_cls

                if weight_fit == False and all_weight_fit == False:
                    loss = 0.7 * 0.5 * loss_seg + 0.5 * loss_cls

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        train_cls_loss += loss_cls.item()
        train_seg_loss += loss_seg.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs_cls, dim=-1), dim=-1) == cls_label).type(torch.FloatTensor))
            train_acc += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'train_cls_loss'   : float(train_cls_loss / (iteration + 1)),
                                'train_seg_loss'   : float(train_seg_loss / (iteration + 1)),
                                'train_acc'        : train_acc / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
 
    if local_rank == 0:
        pbar.close()
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels,cls_label = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                cls_label = cls_label.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs,outputs_cls     = model_train(imgs)
            if focal_loss:
                loss_seg = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                loss_cls = focalLoss(outputs_cls, cls_label)
            else:
                cls_label=cls_label.to(torch.long)
                loss_cls = nn.CrossEntropyLoss()(outputs_cls, cls_label)
                loss_seg = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if weight_fit == True:
                blance = ((epoch + 1)/Epoch)
                loss = (1 - blance) * 0.7 * 0.5 * loss_seg + 0.5 * loss_cls
                
            if all_weight_fit == True:
                loss = (1 - blance) * 0.7 * 0.5 * loss_seg + blance * 0.5 * loss_cls

            if weight_fit == False and all_weight_fit == False:
                loss = 0.7 * 0.5 * loss_seg + 0.5 * loss_cls

            accuracy        = torch.mean((torch.argmax(F.softmax(outputs_cls, dim=-1), dim=-1) == cls_label).type(torch.FloatTensor))

            val_loss    += loss.item()
            val_seg_loss += loss_seg.item()
            val_cls_loss += loss_cls.item()
            val_f_score += _f_score.item()
            val_accuracy    += accuracy.item()
            
            
            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    'val_seg_loss' : val_seg_loss / (iteration + 1),
                                    'val_cls_loss' : val_cls_loss/(iteration + 1),
                                    'accuracy'  : val_accuracy / (iteration + 1), 
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_cls_loss / epoch_step_val)
        #eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || cls Loss: %.3f ' % (total_loss / epoch_step, val_cls_loss / epoch_step_val))
        
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-val_acc%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_cls_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_cls_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
