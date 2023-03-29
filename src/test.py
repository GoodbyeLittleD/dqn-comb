import math
import os
import torch
import numpy as np
import glob

import net

torch.set_printoptions(profile="full")
CHECKPOINT_LOCATION = os.path.join('data', 'checkpoint')
MODEL_LOCATION = os.path.join('data', 'model')
NNARGS = net.NNArgs(
        num_channels = 12,
        depth        = 4,
        lr_milestone = 50,
        dense_net    = True,
        kernel_size  = 5
    )

lines = [[], [], []]
lines[0].append([8, 13, 17])
lines[0].append([4, 9, 14, 18])
lines[0].append([1, 5, 10, 15, 19])
lines[0].append([2, 6, 11, 16])
lines[0].append([3, 7, 12])

lines[1].append([1, 2, 3])
lines[1].append([4, 5, 6, 7])
lines[1].append([8, 9, 10, 11, 12])
lines[1].append([13, 14, 15, 16])
lines[1].append([17, 18, 19])

lines[2].append([1, 4, 8])
lines[2].append([2, 5, 9, 13])
lines[2].append([3, 6, 10, 14, 17])
lines[2].append([7, 11, 15, 18])
lines[2].append([12, 16, 19])

numbers = [[3, 4, 8], [1, 5, 9], [2, 6, 7]]

def get_left(c):
    return 3 if c < 18 else 4 if c < 36 else 8
def get_mid(c):
    return 1 if c % 18 < 6 else 5 if c % 18 < 12 else 9
def get_right(c):
    return 2 if c % 6 < 2 else 6 if c % 6 < 4 else 7

def board_to_feature(board):
    card_list = [
        [0, 0, 0] if i < 0 else [get_left(i), get_mid(i), get_right(i)] 
                  if i < 54 else [10, 10, 10] 
                  for i in board]

    feature = np.zeros(net.CANONICAL_SHAPE, dtype=np.float32)
    # The first feature [i, j] means number (j+1) is in at position i or not.
    # In other words, number j exist on position i, then feature[0, i, j-1] = 1
    # Specially, [10, 10, 10] place will have all feature filled in with 1
    for i in range(20):
        if card_list[i][0] == 10:
            assert card_list[i][1] == 10 and card_list[i][2] == 10
            feature[0, i].fill(1)
        elif card_list[i][0] > 0:
            feature[0, i, card_list[i][0] - 1] = 1
            feature[0, i, card_list[i][1] - 1] = 1
            feature[0, i, card_list[i][2] - 1] = 1
    
    # The second feature [i, j] means number (j+1) is possible to 
    #   get score on its line.
    # In other words, a line has only number k or 10 or empty, then 
    #   feature[1, i, k-1] = 1 for every position i on this line.
    for i in range(1, 20):
        # for each number group, e.g. numbers[j] = 3,4,8
        for j in range(3):
            # for each number
            for k in numbers[j]:
                # test means possible to get score on this line
                test = 1
                # find the line where i lays
                for l in lines[j]:
                    if i in l:
                        # check if position on this line is either:
                        # 1. has number k
                        # 2. is empty
                        for n in l:
                            if feature[0, n, k-1] == 0 and card_list[n][0] != 0:
                                # both condition failed, test fail.
                                test = 0
                                break
                feature[1, i, k-1] = test

    # The third feature simply tells if there is any number on the line.
    # In other words, a line has any number, then for every k in lines[j],
    #   feature[2, i, k-1] = 1 for every position i on this line.
        # for each number group
        for j in range(3):
            # for each number
            for k in numbers[j]:
                # test means nothing on this line
                test = 0
                # find the line where i lies
                for l in lines[j]:
                    if i in l:
                        # check if any number lies on this line:
                        for n in l:
                            if card_list[n][0] != 0:
                                test = 1
                                break
                feature[2, i, k-1] = test
    
    # The fourth feature tells if there is any [10, 10, 10] on the line.
    # In other words, a line has [10, 10, 10], then for every k in lines[j],
    #   feature[3, i, k-1] = 1 for every position i on this line.
        for j in range(3):
            for k in numbers[j]:
                test = 0
                for l in lines[j]:
                    if i in l:
                        for n in l:
                            if card_list[n][0] == 10:
                                test = 1
                                break
                feature[3, i, k-1] = test
    return feature

nn = net.NNWrapper.load_checkpoint(CHECKPOINT_LOCATION, '0060.pt')
nn.nnet.eval()

input_tensor = torch.rand(1, 4, 20, 9).cuda()
traced_script_module = torch.jit.trace(nn.nnet, input_tensor)
traced_script_module.save(os.path.join(MODEL_LOCATION, 'model.pt'))

test_tensor = torch.zeros(1, 4, 20, 9)
test_tensor[0, 0, 0].fill_(1)
test_tensor[0, 1].fill_(1)
test_tensor[0, 1, 0].fill_(0)
#with torch.no_grad():
output = nn.nnet.forward(test_tensor.contiguous().cuda()).item()
print(output)

output = nn.predict(test_tensor.squeeze(0))
print(output)

while True:
    print('Input board:', end=' ')
    board = [int(i) for i in input().strip().split()]
    input_tensor = torch.from_numpy(board_to_feature(board))
    print(input_tensor)
    output = nn.predict(input_tensor).item()
    print(output * 160)

    

