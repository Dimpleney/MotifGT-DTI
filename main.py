# -*- coding: utf-8 -*-
import argparse
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score,roc_curve, precision_recall_curve
from torch.optim.lr_scheduler import ExponentialLR

from dataset import collate_wrapper
from build_dataset import build_dataset
from torch.utils.data import DataLoader
from config import args_to_config
import wandb
from pathlib import Path

from tqdm import tqdm
from model_ban import PerceiverIODTI
import matplotlib.pyplot as plt
import os

def get_args_parser():
    parser = argparse.ArgumentParser('DTIA training and evaluation script', add_help=False)

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/', type=str,
                        help='dataset path')
    parser.add_argument('--raw-data-dir', default='./data/', type=str)
    parser.add_argument('--train-split', default=0.9, type=float)
    parser.add_argument('--val-split', default=0.0, type=float)
    parser.add_argument('--dataset', default='human', choices=['celegans', 'human', 'biosnap', 'drugbank'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--df-dir', default='./data/', type=str)
    parser.add_argument('--processed-file-dir', default='./data/processed/', type=str)
    parser.add_argument('--pdb-dir', default='./data/pdb/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='gpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=4, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    return parser


def get_roce(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce


def test_func(model, dataloader, device, epoch):
    y_pred = []
    y_label = []

    model.eval()
    if not dataloader:
        print("Dataloader is empty.")
        return
    for inp, target in dataloader:
        outs = []
        for item in range(len(inp[0])):
            inpu = (inp[0][item].to(device), inp[1][item].to(device),inp[2][item].to(device))
            out = model(inpu)
            outs.append(out)
        out = torch.stack(outs, dim=0).squeeze(1)
        y_pred.append(out.detach().cpu())
        y_label.append(target.cpu())

    y_pred = torch.cat(y_pred, dim=0)
    y_label = torch.cat(y_label, dim=0)
    y_pred_c = [round(i.item()) for i in y_pred]
    roce1 = get_roce(y_pred, y_label, 0.5)
    roce2 = get_roce(y_pred, y_label, 1)
    roce3 = get_roce(y_pred, y_label, 2)
    roce4 = get_roce(y_pred, y_label, 5)
    auroc = roc_auc_score(y_label, y_pred)
    prauc = average_precision_score(y_label, y_pred)
    f1score = f1_score(y_label, y_pred_c)
    precision = precision_score(y_label, y_pred_c)
    recall = recall_score(y_label, y_pred_c)
    balanced_acc = balanced_accuracy_score(y_label, y_pred_c)

    evaluation_metrics = {
        'AUROC': str(auroc),
        'PRAUC': str(prauc),
        'F1 Score': str(f1score),
        'Precision Score': str(precision),
        'Recall Score': str(recall),
        'Balanced Accuracy Score': str(balanced_acc),
        # 'Alpha Values': str(alpha_values)
    }

    directory = f"./result/{args.dataset}"
    filename = os.path.join(directory, "result.txt")

    with open(filename, 'a') as file:
        file.write(f"Epoch {epoch} Metrics:\n")
        for metric, value in evaluation_metrics.items():
            file.write(f"{metric}: {value}\n")

    print("AUROC: " + str(roc_auc_score(y_label, y_pred)), end=" ")
    print("PRAUC: " + str(average_precision_score(y_label, y_pred)), end=" ")
    print("F1 Score: " + str(f1_score(y_label, y_pred_c)), end=" ")
    print("Precision Score:" + str(precision_score(y_label, y_pred_c)), end=" ")
    print("Recall Score:" + str(recall_score(y_label, y_pred_c)), end=" ")
    print("Balanced Accuracy Score " + str(balanced_accuracy_score(y_label, y_pred_c)), end=" ")
    print("0.5 re Score " + str(roce1), end=" ")
    print("1 re Score " + str(roce2), end=" ")
    print("2 re Score " + str(roce3), end=" ")
    print("5 re Score " + str(roce4), end=" ")
    print("-------------------")

    # AUROC
    fpr, tpr, _ = roc_curve(y_label, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='AUROC (area = %0.2f)' % auroc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, f'roc_curve_epoch_{epoch}.png'))
    plt.close()

    # AUPRC
    precision, recall, _ = precision_recall_curve(y_label, y_pred)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='AUPRC (area = %0.2f)' % prauc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(directory, f'pr_curve_epoch_{epoch}.png'))
    plt.close()


def main(args):
    config = args_to_config(args)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_train, dataset_val, dataset_test = build_dataset(config=config)
    # print("DataLoader_train:")
    data_loader_train = DataLoader(dataset_train, drop_last=False, batch_size=32, shuffle=False,
                                   num_workers=0, pin_memory=False, collate_fn=collate_wrapper)

    data_loader_test = DataLoader(dataset_test, drop_last=False, batch_size=32, collate_fn=collate_wrapper,
                                  num_workers=0, pin_memory=False)
    model = PerceiverIODTI()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.03)
    criterion = torch.nn.BCELoss()
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    epochs = 100
    accum_iter = 2
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        accs = []
        model.train()
        with tqdm(data_loader_train) as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # print(f"DataLoader size: {len(data_loader_train.dataset)}")
            # print(f"Range: {range(len(tepoch))}")
            for batch_idx, (inp, target) in enumerate(tepoch):
                print("batch_idx:",batch_idx)
                # print("inp:", inp)
                # print("target:",target)
                outs = []
                for item in range(len(inp[0])):
                    print("item:", item)
                    # print("inp[0]:", inp[0])
                    inpu = (inp[0][item].to(device), inp[1][item].to(device),inp[2][item].to(device))
                    print("inp_0:", inp[0][item])
                    print("inp_1:", inp[1][item])
                    print("inp_2:", inp[2][item].shape)
                    out = model(inpu)
                    # print("out:", out)
                    outs.append(out)
                out = torch.stack(outs, dim=0).squeeze(1)

                target = target.to(device).view(-1, 1).to(torch.float)
                loss = criterion(out, target)
                matches = [torch.round(i) == torch.round(j) for i, j in zip(out, target)]
                acc = matches.count(True) / len(matches)
                accs.append(acc)
                losses.append(loss.detach().cpu())
                loss.backward()
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(data_loader_train)):
                    optimizer.step()
                    optimizer.zero_grad()

                acc_mean = np.array(accs).mean()
                loss_mean = np.array(losses).mean()
                tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
        # average loss
        epoch_losses.append(np.mean(losses))
        scheduler.step()
        print("test_func:")
        test_func(model, data_loader_test, device,epoch)
        fn = f"last_checkpoint_{args.dataset}.pt"
        info_dict = {
           'epoch': epoch,
           'net_state': model.state_dict(),
           'optimizer_state': optimizer.state_dict()
        }
        torch.save(info_dict, fn)
    # loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, '-o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig(f'./result/{args.dataset}/result.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DTIA training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


