# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)

import logging
from contextlib import nullcontext
# if your python version < 3.7 use the below one
# from contextlib import suppress as nullcontext
import torch
from torch.nn.utils import clip_grad_norm_
import csv
import collections


def check_memory_stat_consistency():
    snapshot = torch.cuda.memory_snapshot()

    expected_each_device = collections.defaultdict(lambda: collections.defaultdict(int))

    # for segment in snapshot:
    for idx, segment in enumerate(snapshot):
        expected = expected_each_device[segment["device"]]
        pool_str = segment["segment_type"] + "_pool"

        # expected["segment.all.current"] += 1
        # expected["segment." + pool_str + ".current"] += 1

        # expected["allocated_bytes.all.current"] += segment["allocated_size"]
        # expected["allocated_bytes." + pool_str + ".current"] += segment["allocated_size"]

        expected["reserved_bytes.all.current"] += segment["total_size"]
        expected["reserved_bytes." + pool_str + ".current"] += segment["total_size"]

        expected["active_bytes.all.current"] += segment["active_size"]
        expected["active_bytes." + pool_str + ".current"] += segment["active_size"]

        is_split = len(segment["blocks"]) > 1
        real = 0.0
        for block in segment["blocks"]:
            # if block["state"] == "active_allocated":
            #     expected["allocation.all.current"] += 1
            #     expected["allocation." + pool_str + ".current"] += 1
            # if idx == 7 or idx == 8:
            print("The block size is:", block["size"] / 1024 / 1024, "MB", ", the block state is:", block["state"])

            if block["state"].startswith("active_"):
                expected["active.all.current"] += 1
                expected["active." + pool_str + ".current"] += 1

            if block["state"] == "inactive" and is_split:
                expected["inactive_split.all.current"] += 1
                expected["inactive_split." + pool_str + ".current"] += 1
                expected["inactive_split_bytes.all.current"] += block["size"]
                expected["inactive_split_bytes." + pool_str + ".current"] += block["size"]
                real += block["size"]
        print("Segment Index:", idx, ", Inactivate split rate: ",
              round((real / segment["total_size"] * 100), 2),
              "%,   Total Size:",
              segment["total_size"] / 1024 / 1024, "MB",
              ",   Inactive Size:",
              round(real / 1024 / 1024, 2), "MB")
    print()

class Executor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer,
              args, scaler):
        ''' Train one epoch
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        rank = args.get('rank', 0)
        accum_grad = args.get('accum_grad', 1)
        is_distributed = args.get('is_distributed', True)
        use_amp = args.get('use_amp', False)
        logging.info('using accumulate grad, new batch size is {} times'
                     'larger than before'.format(accum_grad))
        # if use_amp:
            # assert scaler is not None
        num_seen_utts = 0
        num_total_batch = len(data_loader)
        memory_allocated_list, memory_reserved_list, memory_inactive_list = [], [], []
        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, target_lengths = batch
            if batch_idx % 50 == 0:
                memory_allocated_list.append(torch.cuda.memory_stats()["allocated_bytes.all.current"] / 1024 / 1024)
                memory_reserved_list.append(torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024)
                memory_inactive_list.append(torch.cuda.memory_stats()["inactive_split_bytes.all.current"] / 1024 / 1024)
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if batch_idx % 50 == 0:
                memory_allocated_list.append(torch.cuda.memory_stats()["allocated_bytes.all.current"] / 1024 / 1024)
                memory_reserved_list.append(torch.cuda.memory_stats()["reserved_bytes.all.current"] / 1024 / 1024)
                memory_inactive_list.append(torch.cuda.memory_stats()["inactive_split_bytes.all.current"] / 1024 / 1024)
                # check_memory_stat_consistency()
            num_utts = target_lengths.size(0)
            if num_utts == 0:
                continue
            context = None
            # Disable gradient synchronizations across DDP processes.
            # Within this context, gradients will be accumulated on module
            # variables, which will later be synchronized.
            if is_distributed and batch_idx % accum_grad != 0:
                context = model.no_sync
            # Used for single gpu training and DDP gradient synchronization
            # processes.
            else:
                context = nullcontext
            with context():
                # autocast context
                # The more details about amp can be found in
                # https://pytorch.org/docs/stable/notes/amp_examples.html
                loss, loss_att, loss_ctc = model(feats, feats_lengths,
                                                     target, target_lengths)
                # with torch.cuda.amp.autocast(scaler is not None):
                #     loss, loss_att, loss_ctc = model(feats, feats_lengths,
                #                                      target, target_lengths)
                #     loss = loss / accum_grad
                model.backward(loss)
                # if use_amp:
                #     scaler.scale(loss).backward()
                # else:
                #     loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                if rank == 0 and writer is not None:
                    writer.add_scalar('train_loss', loss, self.step)
                # Use mixed precision training
                model.step()
                # if use_amp:
                #     scaler.unscale_(optimizer)
                #     grad_norm = clip_grad_norm_(model.parameters(), clip)
                #     # Must invoke scaler.update() if unscale_() is used in the
                #     # iteration to avoid the following error:
                #     #   RuntimeError: unscale_() has already been called
                #     #   on this optimizer since the last update().
                #     # We don't check grad here since that if the gradient has
                #     # inf/nan values, scaler.step will skip optimizer.step().
                #     scaler.step(optimizer)
                #     scaler.update()
                # else:
                #     grad_norm = clip_grad_norm_(model.parameters(), clip)
                #     if torch.isfinite(grad_norm):
                #         optimizer.step()
                # optimizer.zero_grad()
                # scheduler.step()
                self.step += 1
            if batch_idx % log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    batch_idx, num_total_batch,
                    loss.item() * accum_grad)
                if loss_att is not None:
                    log_str += 'loss_att {:.6f} '.format(loss_att.item())
                if loss_ctc is not None:
                    log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                log_str += 'lr {:.8f} rank {}'.format(lr, rank)
                logging.debug(log_str)
        with open('test.csv', 'w') as f:
            write = csv.writer(f)
            write.writerow(memory_allocated_list)
            write.writerow(memory_reserved_list)
            write.writerow(memory_inactive_list)

    def cv(self, model, data_loader, device, args):
        ''' Cross validation on
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        num_total_batch = len(data_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                loss, loss_att, loss_ctc = model(feats, feats_lengths, target,
                                                 target_lengths)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                if batch_idx % log_interval == 0:
                    log_str = 'CV Batch {}/{} loss {:.6f} '.format(
                        batch_idx, num_total_batch, loss.item())
                    if loss_att is not None:
                        log_str += 'loss_att {:.6f} '.format(loss_att.item())
                    if loss_ctc is not None:
                        log_str += 'loss_ctc {:.6f} '.format(loss_ctc.item())
                    log_str += 'history loss {:.6f}'.format(total_loss /
                                                            num_seen_utts)
                    logging.debug(log_str)

        return total_loss, num_seen_utts
