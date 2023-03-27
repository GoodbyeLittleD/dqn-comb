from collections import namedtuple
import os
import torch
from torch import optim, nn
from torch.autograd import profiler
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm
import numpy as np

# This is an autotuner for network speed.
torch.backends.cudnn.benchmark = True

NNArgs = namedtuple('NNArgs', ['num_channels', 'depth', 'kernel_size', 'lr_milestone', 'dense_net',
                               'lr', 'cv', 'cuda'], defaults=(40, False, 0.01, 1.5, torch.cuda.is_available()))

CANONICAL_SHAPE = (3, 20, 9)

def conv(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding='same', bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 1)


def conv3x3(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 3)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(in_channels, growth_rate*bn_size, 1)
        self.bn2 = nn.BatchNorm2d(growth_rate*bn_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(growth_rate*bn_size, growth_rate,
                          kernel_size=kernel_size)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, kernel_size=3):
        super(ResidualBlock, self).__init__()
        stride = 1
        if downsample:
            stride = 2
            self.conv_ds = conv1x1(in_channels, out_channels, stride)
            self.bn_ds = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(in_channels, out_channels,
                          stride, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.conv_ds(x)
            residual = self.bn_ds(residual)
        out += residual
        return out


class NNArch(nn.Module):
    def __init__(self, args):
        super(NNArch, self).__init__()
        in_channels, in_x, in_y = CANONICAL_SHAPE
        self.dense_net = args.dense_net

        if not self.dense_net:
            self.conv1 = conv(in_channels, args.num_channels,
                              kernel_size=args.kernel_size)
            self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.layers = []
        for i in range(args.depth):
            if self.dense_net:
                self.layers.append(DenseBlock(
                    in_channels + args.num_channels*i, args.num_channels, kernel_size=args.kernel_size))
            else:
                self.layers.append(ResidualBlock(
                    args.num_channels, args.num_channels, kernel_size=args.kernel_size))
        self.conv_layers = nn.Sequential(*self.layers)

        if self.dense_net:
            final_size = in_channels + args.num_channels * args.depth
            self.v_conv = conv1x1(final_size, 32)
        else:
            self.v_conv = conv1x1(args.num_channels, 32)

        self.v_bn = nn.BatchNorm2d(32)
        self.v_relu = nn.ReLU(inplace=True)
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(32*in_x*in_y,
                               256)
        self.v_fc1_relu = nn.ReLU(inplace=True)
        self.v_fc2 = nn.Linear(256, 1)
        #self.v_softmax = nn.LogSoftmax(1)

    def forward(self, s):
        with profiler.record_function("conv-layers"):
            if not self.dense_net:
                s = self.conv1(s)
                s = self.bn1(s)
            #print('step 0', s)
            s = self.conv_layers(s)
            #print('step 1', s)

        with profiler.record_function("v-head"):
            v = self.v_conv(s)
            v = self.v_bn(v)
            #print('step 2', v)
            v = self.v_relu(v)
            v = self.v_flatten(v)
            #print('step 3', v)
            v = self.v_fc1(v)
            v = self.v_fc1_relu(v)
            #print('step 4', v)
            v = self.v_fc2(v)
            #v = self.v_softmax(v)
            #print('step 5', v)

        return v


class NNWrapper:
    def __init__(self, args):
        self.args = args
        self.nnet = NNArch(args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

        def lr_lambda(epoch):
            if epoch < 5:
                return 1/30
            elif epoch > args.lr_milestone:
                return 3/min(epoch, 600)         # after epoch 150, lr starts to decrease from 0.2 to a minimum of 0.05
            else:
                return 0.1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda)
        # self.scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, milestones=args.lr_milestones, gamma=0.1)
        self.cuda = args.cuda
        self.cv = args.cv
        if self.cuda:
            self.nnet.cuda()

    def losses(self, dataset):
        self.nnet.eval()
        l_v = 0
        for batch in tqdm(dataset, desc='Calculating Sample Loss', leave=False):
            canonical, target_vs = batch
            if self.cuda:
                canonical = canonical.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()

            out_v = self.nnet(canonical)
            l_v += self.loss_v(target_vs, out_v).item()
        return l_v/len(dataset)

    def sample_loss(self, dataset, size):
        loss = np.zeros(size)
        self.nnet.eval()
        i = 0
        for batch in tqdm(dataset, desc='Calculating Sample Loss', leave=False):
            canonical, target_vs = batch
            if self.cuda:
                canonical = canonical.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()

            out_v = self.nnet(canonical)
            total_loss = self.sample_loss_v(target_vs, out_v)
            for sample_loss in total_loss:
                loss[i] = sample_loss
                i += 1
        return loss

    def train(self, batches, steps_to_train, run, epoch, total_train_steps):
        self.nnet.train()

        v_loss = 0
        current_step = 0
        pbar = tqdm(total=steps_to_train, unit='batches',
                    desc='Training NN', leave=False)
        past_states = []
        while current_step < steps_to_train:
            for batch in batches:
                if steps_to_train//4 > 0 and current_step % (steps_to_train//4) == 0 and current_step != 0:
                    # Snapshot model weights
                    past_states.append(dict(self.nnet.named_parameters()))
                if current_step == steps_to_train:
                    break
                canonical, target_vs = batch
                if self.cuda:
                    canonical = canonical.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v = self.nnet(canonical)
                # print(target_vs)
                # print(out_v)
                l_v = self.loss_v(target_vs, out_v)
                # print(l_v)
                total_loss = l_v
                total_loss.backward()
                self.optimizer.step()

                run.track(l_v.item(), name='loss', epoch=epoch, step=total_train_steps+current_step,
                          context={'type': 'value'})

                # record loss and update progress bar.
                v_loss += l_v.item()
                current_step += 1
                pbar.set_postfix({'loss': v_loss/current_step})
                pbar.update()

        # Perform expontential averaging of network weights.
        past_states.append(dict(self.nnet.named_parameters()))
        merged_states = past_states[0]
        for state in past_states[1:]:
            for k in merged_states.keys():
                merged_states[k].data = merged_states[k].data * \
                    0.75 + state[k].data * 0.25
        nnet_dict = self.nnet.state_dict()
        nnet_dict.update(merged_states)
        self.nnet.load_state_dict(nnet_dict)

        self.scheduler.step()
        pbar.close()
        return v_loss/steps_to_train

    def predict(self, canonical):
        v = self.process(canonical.unsqueeze(0))
        return v[0]

    def process(self, batch):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        if self.cuda:
            batch = batch.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            v = self.nnet(batch)
            res = torch.exp(v)
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        return res

    def sample_loss_v(self, targets, outputs):
        return -self.cv * torch.sum(targets * outputs, axis=1)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]
        # return -self.cv * torch.sum(targets * outputs) / targets.size()[0]

    def save_checkpoint(self, folder=os.path.join('data','checkpoint'), filename='checkpoint.pt'):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save({
            'state_dict': self.nnet.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'sch_state': self.scheduler.state_dict(),
            'args': self.args
        }, filepath)

    @staticmethod
    def load_checkpoint(folder=os.path.join('data','checkpoint'), filename='checkpoint.pt'):
        if folder != '':
            filepath = os.path.join(folder, filename)
        else:
            filepath = filename
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")
        checkpoint = torch.load(filepath)
        net = NNWrapper(checkpoint['args'])
        net.nnet.load_state_dict(checkpoint['state_dict'])
        net.optimizer.load_state_dict(checkpoint['opt_state'])
        net.scheduler.load_state_dict(checkpoint['sch_state'])
        return net

