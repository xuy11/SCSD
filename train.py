import argparse
import os.path as osp
import os
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from model import Net
import utils
from utils import AverageMeter,Init_random_seed,get_param_groups,get_config,generate_hidden_dim
import MLdataset
from loss import Loss
import evaluation


def train(loader, model, loss_model, opt, sche, epoch,last_preds,logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    All_preds = torch.tensor([]).cuda()

    model.quantize.reset_cluster_size('cuda:0')

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data = [v_data.to('cuda:0') for v_data in data]

        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')

        xrec_list, pred, loss_quantize, z_list, pred_list = model(data,mask=inc_V_ind)


        All_preds = torch.cat([All_preds,pred],dim=0)

        loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind)
        teacher_pred = pred.detach()
        loss_self_distill = 0.0
        for v in range(len(pred_list)):
            valid_mask = inc_V_ind[:, v].unsqueeze(-1)
            if valid_mask.sum() == 0:
                continue
            student_pred = pred_list[v]
            distill_loss = F.binary_cross_entropy(student_pred, teacher_pred, reduction="none")
            distill_loss = (distill_loss * valid_mask).sum() / (valid_mask.sum() + 1e-9)
            student_loss = loss_model.weighted_BCE_loss(student_pred, label, inc_L_ind, reduction='none')
            fuse_mask = valid_mask * inc_L_ind
            student_loss = (student_loss * fuse_mask).sum() / (fuse_mask.sum() + 1e-9)
            loss_self_distill += args.lambd * distill_loss + (1 - args.lambd) * student_loss

        loss_mse = 0
        for v in range(len(data)):
            for j in range(len(data)):
                loss_mse += loss_model.weighted_wmse_loss_sum(data[j], xrec_list[v][j], (inc_V_ind[:, v].int() & inc_V_ind[:, j].int()).float(), reduction='mean')
        loss_mse = torch.mean(loss_mse / torch.sum(inc_V_ind, dim=1))

        loss = loss_CL + loss_mse * args.alpha + loss_quantize + loss_self_distill

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

        opt.step()

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model,All_preds

def test(loader, model, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model.eval()
    end = time.time()

    model.quantize.reset_cluster_size('cuda:0')

    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data = [v_data.to('cuda:0') for v_data in data]

        _, pred, _, _, _ = model(data, mask=inc_V_ind.to('cuda:0'))

        pred = pred.cpu()
        total_labels = np.concatenate((total_labels, label.numpy()), axis=0) if len(total_labels) > 0 else label.numpy()
        total_preds = np.concatenate((total_preds, pred.detach().numpy()), axis=0) if len(total_preds) > 0 else pred.detach().numpy()

        loss = loss_model.weighted_BCE_loss(pred, label, inc_L_ind)

        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    total_labels = np.array(total_labels)
    total_preds = np.array(total_preds)

    codebook_cluster_size = model.quantize.cluster_size
    zero_cnt = (codebook_cluster_size == 0).sum().item()
    print(f"Unused code in codebook(unused,used,total): {zero_cnt,len(codebook_cluster_size)-zero_cnt,len(codebook_cluster_size)}")

    evaluation_results = evaluation.do_metric(total_preds, total_labels)
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}\t'
                'AP {ap:.3f}\t'
                'HL {hl:.3f}\t'
                'RL {rl:.3f}\t'
                'AUC {auc:.3f}\t'.format(
        epoch, batch_time=batch_time,
        losses=losses,
        ap=evaluation_results[0],
        hl=evaluation_results[1],
        rl=evaluation_results[2],
        auc=evaluation_results[3]
    ))
    return evaluation_results

def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset + '_six_view_MaskRatios_' + str(
        args.mask_view_ratio) + '_LabelMaskRatio_' +
                              str(args.mask_label_ratio) + '_TraindataRatio_' +
                              str(args.training_sample_ratio) + '.mat')

    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir, args.name + args.dataset + '_V_' + str(
            args.mask_view_ratio) + '_L_' +
                           str(args.mask_label_ratio) + '_T_' +
                           str(args.training_sample_ratio)+ '_' + str(args.lr) + '_' + str(args.alpha) + '_' + str(args.lambd)+ '.txt')
    else:
        logfile = None
    logger = utils.setLogger(logfile)
    device = torch.device('cuda:0')
    for fold_idx in range(folds_num):
        fold_idx = fold_idx
        train_dataloder, train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx, mode='train',batch_size=args.batch_size, shuffle=True,num_workers=4)
        test_dataloder, test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15, fold_idx=fold_idx, mode='test',batch_size=args.batch_size, num_workers=4)
        val_dataloder, val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx, mode='val',batch_size=args.batch_size, num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')

        dep_graph = torch.matmul(labels.T, labels)
        dep_graph = dep_graph / (torch.diag(dep_graph).unsqueeze(1) + 1e-10)
        dep_graph.fill_diagonal_(fill_value=0.)
        dep_graph = (dep_graph + dep_graph.T) / 2
        dep_graph = dep_graph / (dep_graph.sum(dim=1, keepdim=True) + 1e-9)


        model = Net(d_list, classes_num, args.z_dim, args.hidden_dim, args.num_tokens, args.codebook_dim, args.decay, args.dropout_rate,dep_graph,args.tau)
        model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


        loss_model = Loss()

        optimizer = AdamW(get_param_groups(model, weight_decay=args.weight_decay), lr=args.lr)

        scheduler = None

        logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(
            len(test_dataset)) + '   fold_idx:' + str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        best_epoch = 0
        best_model_dict = {'model': model.state_dict(), 'epoch': 0}

        for epoch in range(args.epochs):
            if epoch==0:
                All_preds = None

            if epoch < args.warmup_epoch:
                lr_scale = (epoch + 1) / args.warmup_epoch
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * lr_scale

            train_losses,model,All_preds = train(train_dataloder,model,loss_model,optimizer,scheduler,epoch,All_preds,logger)

            val_results = test(val_dataloder, model, loss_model, epoch, logger)

            if val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.25 >= static_res:
                static_res = val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.25
                best_model_dict['model'] = copy.deepcopy(model.state_dict())
                best_model_dict['epoch'] = epoch
                best_epoch = epoch
            total_losses.update(train_losses.sum)

            if scheduler is not None and epoch >= args.warmup_epoch:
                scheduler.step()

            if (epoch - best_epoch > args.patience):
                print('Training stopped: epoch=%d' % (epoch))
                break

        model.load_state_dict(best_model_dict['model'])
        print("epoch", best_model_dict['epoch'])
        test_results = test(test_dataloder, model, loss_model, epoch, logger)
        logger.info('final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(
                fold_idx, best_epoch, test_results[0], test_results[1],test_results[2], test_results[3]))
        for i in range(9):
            folds_results[i].update(test_results[i])

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write('AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha lambd\n')
    res_list = [str(round(res.avg, 3)) + '+' + str(round(res.std, 3)) for res in folds_results]
    res_list.extend([str(args.lr), str(args.alpha), str(args.lambd)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH', default='')
    parser.add_argument('--config-path', type=str, metavar='PATH',  default=osp.join(working_dir, 'config'))
    parser.add_argument('--root-dir', type=str, metavar='PATH', default='data/')
    parser.add_argument('--dataset', type=str, default='corel5k')
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'weights'))
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)

    parser.add_argument('--name', type=str, default='final_')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=3)
    parser.add_argument('--num_tokens', type=int, default=2048)
    parser.add_argument('--codebook_dim', type=int, default=4)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--tau', type=float, default=5.0)

    parser.add_argument('--lambd', type=float, default=1e-1)

    args = parser.parse_args()
    args = get_config(args, osp.join(args.config_path, args.dataset + '.yaml'))
    args.hidden_dim = generate_hidden_dim(args.hidden_size, args.z_dim)
    Init_random_seed(args.seed)

    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)

    file_path = osp.join(args.records_dir, args.name + args.dataset + '_VM_' + str(args.mask_view_ratio) + '_LM_' +
                         str(args.mask_label_ratio) + '_T_' +
                         str(args.training_sample_ratio) + '.txt')
    args.file_path = file_path

    main(args, file_path)