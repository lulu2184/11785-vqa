import errno
import os
import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VQADataset
from expr_parsing_model import build_expr_parsing_attention_model
from preprocessing.word_dictionary import WordDict

BATCH_SIZE = 512
HIDDEN_NUMBER = 1024
NUMBER_OF_WORKERS = 8


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def inference(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v_dev, q_dev, a_dev in iter(dataloader):
        v_dev = Variable(v_dev).cuda()
        q_dev = Variable(q_dev).cuda()
        pred = model(v_dev, q_dev, None)
        batch_score = compute_score_with_logits(pred, a_dev.cuda()).sum()
        score += batch_score
        upper_bound += (a_dev.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


def train(model, train_loader, dev_loader, num_epochs=30, output_dir='output'):
    print('Start Training...')
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                raise

    writer = SummaryWriter()
    optim = torch.optim.Adamax(model.parameters())
    best_dev_score = 0
    log_file = open("log.txt", "w")

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

            if i % 10 == 0:
                print("epoch:{0}, batch {1}/{2}.".format(epoch + 1, i + 1,
                                                         len(train_loader)))

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        dev_score, bound = inference(model, dev_loader)
        model.train(True)

        epoch_statistics1 = 'epoch {0}, time: {1:.2f}'.format(epoch,
                                                              time.time() - t)
        epoch_statistics2 = '\ttrain_loss: {0:.2f}, score: {1:.2f}'.format(
            total_loss, train_score)
        epoch_statistics3 = '\tdev score: {0:.2f}/{1:.2f}'.format(
            100 * dev_score, 100 * bound)
        print(epoch_statistics1)
        print(epoch_statistics2)
        print(epoch_statistics3)

        log_file.write(epoch_statistics1)
        log_file.write(epoch_statistics2)
        log_file.write(epoch_statistics3)
        log_file.write('\n')
        log_file.flush()

        if dev_score > best_dev_score:
            model_path = os.path.join(output_dir, 'model-{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path)
            best_dev_score = dev_score


if __name__ == '__main__':
    dictionary = WordDict.load_from_file('data/dictionary.pkl')
    train_dataset = VQADataset('train', dictionary)
    dev_dataset = VQADataset('val', dictionary)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True,
                              num_workers=NUMBER_OF_WORKERS)
    dev_loader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=True,
                            num_workers=NUMBER_OF_WORKERS)
    # model = build_baseline0(train_dataset, HIDDEN_NUMBER)
    # model = build_multi_head_attention_model(train_dataset, HIDDEN_NUMBER)
    model = build_expr_parsing_attention_model(train_dataset, HIDDEN_NUMBER)
    model = nn.DataParallel(model).cuda()

    train(model, train_loader, dev_loader)
