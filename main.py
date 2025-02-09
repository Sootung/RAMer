from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse     
from models.models import *
from models.optimization import BertAdam
from utils.eval import get_metrics
from torch.utils.data import DataLoader

from utils.util import get_logger, create_dataloader
from dataloaders.cmu_dataloader import AlignedMoseiDataset, UnAlignedMoseiDataset, M3EDDataset
import pdb


# Move the assignment of logger before the global declaration
# logger = None
global logger
def get_args(description='Multi-modal Multi-label Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", action='store_true', help="whether to run test")
    parser.add_argument("--aligned", action='store_true', help="whether train align of unalign dataset")
    parser.add_argument("--data_path", default='./data/train_valid_test.pt', type=str, help='cmu_mosei data_path')
    parser.add_argument("--output_dir", default='./model_saved/', type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit') 
    parser.add_argument('--unaligned_data_path', type=str, default='/data/mosei_senti_data_noalign.pkl', help='load unaligned dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') 
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--text_dim', type=int, default=300, help='text_feature_dimension') 
    parser.add_argument('--video_dim', type=int, default=35, help='video feature dimension')
    parser.add_argument('--audio_dim', type=int, default=74, help='audio_feature_dimension') 
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--id', type=int, default=0, help='random seed')
    parser.add_argument("--text_model", default="configs", type=str, required=False, help="text module")
    parser.add_argument("--visual_model", default="configs", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="configs", type=str, required=False, help="Audio module")
    parser.add_argument("--personality_model", default="configs", type=str, required=False, help="personality module")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--text_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument("--num_classes", default=6, type=int, required=False)
    parser.add_argument("--hidden_size",type=int, default=256)
    parser.add_argument("--proj_size", type=int, default=64)
    parser.add_argument('--gpu_id', default='0', type=str)
    parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
    parser.add_argument('--lsr_clf_weight', type=float, default=0.01)
    parser.add_argument('--recon_clf_weight', type=float, default=1.0) #1.0
    parser.add_argument('--aug_clf_weight', type=float, default=0.1)
    parser.add_argument('--beta_clf_weight', type=float, default=1.0)
    parser.add_argument('--cl_weight', type=float, default=1.0)
    parser.add_argument('--aug_mse_weight', type=float, default=1.0)
    parser.add_argument('--beta_mse_weight', type=float, default=1.0)
    parser.add_argument('--recon_mse_weight', type=float, default=1.0)
    parser.add_argument('--lsr_vae_weight', type=float, default=1.0)
    parser.add_argument('--total_aug_clf_weight', type=float, default=1.0)
    parser.add_argument('--shuffle_aug_clf_weight', type=float, default=1.0)
    parser.add_argument('--com_pri_weight', type=float, default=0.01)
    parser.add_argument('--diff_weight', type=float, default=5e-6)
    parser.add_argument('--cml_weight', type=float, default=0.5)
    parser.add_argument('--moco_queue', type=int, default=8192)
    parser.add_argument('--binary_threshold', type=float, default=0.35)
    parser.add_argument('--unaligned_mask_same_length', action='store_true')
    parser.add_argument('--save_sub_tensor', action='store_true')
    parser.add_argument('--save_sub_tensor_rec', action='store_true')
    parser.add_argument('--comp_adv_loss', action='store_true')
    parser.add_argument('--comp_rec_loss', action='store_true')



    args = parser.parse_args()
    # Check paramenters
    if args.gradient_accumulation_steps < 1: 
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    if args.aligned:
        output_path = os.path.join(args.output_dir, 'aligned')
    else:
        output_path = os.path.join(args.output_dir, 'unaligned')
        if args.unaligned_mask_same_length:
            output_path = os.path.join(output_path, 'same_length')
        else:
            output_path = os.path.join(output_path, 'not_same_length')
    output_path = os.path.join(output_path, 'ramer_sd{}_t{}_id{}'.format(args.seed, args.binary_threshold, args.id))
    args.output_dir = output_path
    return args


def set_seed_logger(args): 
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    torch.cuda.set_device(args.local_rank) 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    return args


def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu
    if args.batch_size % args.n_gpu != 0: 
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
    return device, n_gpu


def prep_optimizer(args, model, num_train_optimization_steps):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    return optimizer, scheduler, model

def prep_dataloader(args):
    if 'm3ed' in args.data_path:
        Dataset = M3EDDataset
    else:
        Dataset = AlignedMoseiDataset if args.aligned else UnAlignedMoseiDataset
    train_dataset = Dataset(
        args.data_path,
        'train',
        args
    )
    val_dataset = Dataset(
        args.data_path,
        'valid',
        args
    )
    test_dataset = Dataset(
        args.data_path,
        'test',
         args
    )
    # text = train_dataset._get_text(0)
    # visual = train_dataset._get_visual(0)
    # audio = train_dataset._get_audio(0)
    # labels = train_dataset._get_labels(0)
    # print(f"text: {text}, visual: {visual}, audio: {audio}, labels: {labels}")
    # print(f"text dimension: {text[0].shape}, visual dimension: {visual[0].shape}, audio dimension: {audio[0].shape}, labels dimension: {labels.shape}")
    # pdb.set_trace()
    label_input, label_mask = train_dataset._get_label_input()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=True   
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        # drop_last=True
    )
    return train_dataloader, val_dataloader, test_dataloader, label_input, label_mask



def save_model(args, model, epoch):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model_{}.bin.".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(args, n_gpu, device, model_file=None):
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        model = RAMer.from_pretrained(args.text_model, args.visual_model, args.audio_model, args.personality_model,
                                       state_dict=model_state_dict, task_config=args)
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0, label_input=None, label_mask=None):
    global logger
    model.train()
    log_step = args.n_display
    total_loss = 0
    total_pred = []
    total_true_label = []
    total_pred_scores = []

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        pairs_text, pairs_mask, video, video_mask, audio, audio_mask, ground_label = batch
        # ground_label, visual,audio,text,personality, visual_mask, audio_mask,text_mask, personality_mask, target_loc, umask, seg_len, n_c = batch
        # print(f'current batch_idx: {step},current data: {batch}')
        # pdb.set_trace()
        # seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        model_loss, batch_pred, true_label, pred_scores = model(video,audio,pairs_text,video_mask,audio_mask,pairs_mask, groundTruth_labels=ground_label, training=True)
        if n_gpu > 1:
            model_loss = model_loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            model_loss = model_loss / args.gradient_accumulation_steps
        model_loss.backward()
        total_loss += float(model_loss)
        total_pred.append(batch_pred)
        total_true_label.append(true_label)
        total_pred_scores.append(pred_scores)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(model_loss))
    total_loss = total_loss / len(train_dataloader)
    total_pred = torch.cat(total_pred, 0)
    total_true_label = torch.cat(total_true_label, 0)
    total_pred_scores = torch.cat(total_pred_scores, 0)
    return total_loss, total_pred, total_true_label, total_pred_scores


def eval_epoch(args, model, val_dataloader, device, n_gpu, label_input, label_mask, do_save=False):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    with torch.no_grad():
        total_pred = []
        total_true_label = []
        total_pred_scores = []
        total_save_tensor = []
        total_max_idx_1 = []
        total_max_idx_3 = []
        total_max_idx_4 = []
        for _, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # ground_label, visual,audio,text,personality, visual_mask, audio_mask,text_mask, personality_mask, target_loc, umask, seg_len, n_c = batch
            # seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            # return_list = model(visual,audio,text,personality, visual_mask, audio_mask,text_mask, personality_mask,
            #                                                     seq_lengths, target_loc, seg_len, n_c, groundTruth_labels=ground_label, training=False)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels = batch
            return_list = model(video,audio,text, video_mask, audio_mask,text_mask,  
                                                        label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False)
            # batch_pred, true_label, pred_scores = return_list
            if args.save_sub_tensor:
                batch_pred, true_label, pred_scores, norm_tensor, max_idx_list = return_list
                total_save_tensor.append(norm_tensor)
                # print(norm_tensor.shape)
                total_max_idx_1.append(max_idx_list[0])
                total_max_idx_3.append(max_idx_list[1])
                total_max_idx_4.append(max_idx_list[2])
            else:
                batch_pred, true_label, pred_scores = return_list
            total_true_label.append(true_label)
            total_pred.append(batch_pred)
            total_pred_scores.append(pred_scores)
        total_pred = torch.cat(total_pred, 0)
        total_true_label = torch.cat(total_true_label, 0)
        if do_save and args.save_sub_tensor:
            total_save_tensor = torch.cat(total_save_tensor, 0)
            # print(total_save_tensor.shape)
            # print(total_true_label.shape)

            total_max_idx_1 = torch.cat(total_max_idx_1, 0)
            total_max_idx_3 = torch.cat(total_max_idx_3, 0)
            total_max_idx_4 = torch.cat(total_max_idx_4, 0)
            print(total_max_idx_1.shape)
            gt_label_path = os.path.join(args.output_dir, 'gt_labels_wo_adv.pt')
            sub_tensor_path = os.path.join(args.output_dir, 'sub_tensor_wo_adv.pt')
            max_1_path = os.path.join(args.output_dir, 'max_1_wo_adv.pt')
            torch.save(total_max_idx_1.cpu(), max_1_path)
            max_3_path = os.path.join(args.output_dir, 'max_3_wo_adv.pt')
            torch.save(total_max_idx_3.cpu(), max_3_path)
            max_4_path = os.path.join(args.output_dir, 'max_4_wo_adv.pt')
            torch.save(total_max_idx_4.cpu(), max_4_path)
            torch.save(total_true_label.cpu(), gt_label_path)
            torch.save(total_save_tensor.cpu(), sub_tensor_path)
        total_pred_scores = torch.cat(total_pred_scores, 0)
        return total_pred, total_true_label, total_pred_scores
           
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    model = RAMer.from_pretrained(args.text_model, args.visual_model, args.audio_model, args.personality_model,
                                       task_config=args)
    # print the content of args
    # print("the content of args:",args)
    
    # import pdb
    # pdb.set_trace()
    model = model.to(device)

    if args.do_train:
        train_dataloader, val_dataloader, test_dataloader, label_input, label_mask = prep_dataloader(args)
        # data_loader = create_dataloader(args)
        # valid_data_loader = data_loader.split_validation()
        # print(f"train data_loader: {len(train_dataloader)}", f"valid data_loader: {len(val_dataloader)}")
        # pdb.set_trace()
        label_input = label_input.to(device)
        label_mask = label_mask.to(device)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        # print(f"current model: {model}", f"current optimizer: {optimizer}", f"current scheduler: {scheduler}, device: {device}")
        best_score = 0.000
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            total_loss, total_pred, total_label, total_pred_scores = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                scheduler, global_step, local_rank=args.local_rank, label_input=label_input, label_mask=label_mask)
            if args.local_rank == 0:
                logger.info("Epoch %d/%d Finished, Train Loss: %f.",
                            epoch + 1, args.epochs, total_loss)

            total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
            if args.local_rank == 0:
                logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f.", total_micro_f1, total_micro_precision, total_micro_recall, total_acc)
            if args.local_rank == 0:
                logger.info("***** Running valing *****")
                val_pred, val_label, val_pred_scores = eval_epoch(args, model, val_dataloader, device, n_gpu,
                                                                  label_input, label_mask)
                val_micro_f1, val_micro_precision, val_micro_recall, val_acc = get_metrics(val_pred, val_label)
                comp_score = val_micro_f1
                logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f.",
                            val_micro_f1, val_micro_precision, val_micro_recall, val_acc)
                output_model_file = save_model(args, model, epoch)
                if best_score <= comp_score:
                    best_score = comp_score
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the f1 is: {:.4f}".format(best_output_model_file, best_score))
        if args.local_rank == 0:
            logger.info('***** Running testing *****')
            best_model = load_model(args, n_gpu, device, model_file=best_output_model_file)
            if args.save_sub_tensor:
                test_pred, test_label, test_pred_scores = eval_epoch(args, best_model, test_dataloader,
                                                                               device, n_gpu, label_input, label_mask, do_save=True)
            else:
                test_pred, test_label, test_pred_scores = eval_epoch(args, best_model, test_dataloader,
                                                                            device, n_gpu, label_input, label_mask)
            test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(test_pred, test_label)
            logger.info(" res: f1 %f,\tp %f,\tr %f,\tacc %f",
                        test_micro_f1, test_micro_precision, test_micro_recall, test_acc)

      
if __name__ == "__main__":
    main()



