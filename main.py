"""Main game logic for 2048"""

import numpy as np
from copy import copy, deepcopy
NULLVAL = 0
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
actions = [UP, DOWN, LEFT, RIGHT]
dimen = 4

def mergeSeq(seq):
    # merge from left
    # this is not exhaustive merge. can change to merge everything.
    csr = 0
    reward = 0
    while csr + 1 <= len(seq) - 1:
        if seq[csr] == seq[csr+1]:
            seq[csr+1] = seq[csr] * 2
            reward += seq[csr+1]
            seq.pop(csr)
            # csr += 1 # comment this out give an additional merge to
        else:
            csr += 1
    return seq, reward

def fallRow(row, dir=LEFT):
    seq = [num for num in row if num != NULLVAL]
    newrow = [NULLVAL] * 4
    if dir == LEFT:
        seq, reward = mergeSeq(seq)
        for i, num in enumerate(seq):
            newrow[i] = num
    elif dir == RIGHT:
        seq = seq[::-1]
        seq, reward = mergeSeq(seq)
        for i, num in enumerate(seq):
            newrow[-i-1] = num
    else:
        raise ValueError
    return newrow, reward

def fallBoard(board, dir=LEFT):
    reward_all = 0
    if dir == LEFT:
        for i in range(dimen):
            row = board[i, :]
            newrow, reward = fallRow(row, dir=LEFT)
            board[i, :] = newrow
            reward_all += reward
    elif dir == RIGHT:
        for i in range(dimen):
            row = board[i, :]
            newrow, reward = fallRow(row, dir=RIGHT)
            board[i, :] = newrow
            reward_all += reward
    if dir == UP:
        for i in range(dimen):
            row = board[:, i]
            newrow, reward = fallRow(row, dir=LEFT)
            board[:, i] = newrow
            reward_all += reward
    elif dir == DOWN:
        for i in range(dimen):
            row = board[:, i]
            newrow, reward = fallRow(row, dir=RIGHT)
            board[:, i] = newrow
            reward_all += reward

    return board, reward_all

#%%
def getSuccessor(state, action, show=True, clone=True):
    if clone: state = copy(state)
    nextstate, reward = fallBoard(state, action)
    emptypos = getEmptyPos(nextstate)
    if len(emptypos) == 0:
        finished = True
    else:
        nextstate = addRandomFall(nextstate, 2)
        finished = False
    if show: print(nextstate)
    return nextstate, reward, finished

def getSuccessors(state, action, clone=True):
    if clone: state = copy(state)
    nextstate, reward = fallBoard(state, action)
    emptypos = getEmptyPos(nextstate)
    if len(emptypos) == 0:
        finished = True
        nextstates = [nextstate]
    else:
        nextstates = []
        for pos in emptypos:
            nextstate_ = copy(nextstate)
            nextstate_[pos] = 2
            nextstates.append(nextstate_)
        finished = False
    return nextstates, reward, finished

from random import choice, sample
def addRandomFall(state, val):
    posList = []
    for i in range(dimen):
        for j in range(dimen):
            if state[i, j] == NULLVAL:
                posList.append((i, j))
    # posList = getEmptyPos(state)
    rndpos = choice(posList)
    state[rndpos] = val
    return state

def getEmptyPos(state):
    posList = []
    for i in range(dimen):
        for j in range(dimen):
            if state[i, j] == NULLVAL:
                posList.append((i, j))
    return posList

def getInitState():
    board = NULLVAL * np.ones((4, 4), dtype=np.int)
    addRandomFall(board, 2)
    return board

#%% Collection of baseline policies.
def RandomPolicy(board):
    act = choice(actions)
    return act

def RndMax(board, level=4):
    """"""
    bestVal = -1E6
    bestAct = None
    for act in actions:
        nextboard, reward, finished = getSuccessor(board, action=act, show=False, clone=True)
        if level > 0 and not finished:
            nextvalue, nextact = ExpectiMax(nextboard, level=level-1)
        else:
            nextvalue = 0 # evaluation function!
        curvalue = nextvalue + reward - (finished) * 1000 # punishment for death
        if curvalue > bestVal: bestVal, bestAct = curvalue, act
    return bestAct, bestVal

def ExpectiMax(board, level=4, sampn=5):
    """ExpectiMax policy (Expectation over random fall of blocks), with certain depth termination.
    sampln determine the number """
    bestVal = -1E6
    bestAct = None
    for act in actions:
        nextboards, reward, finished = getSuccessors(board, action=act, clone=True)
        if level > 0 and not finished:
            value_col = []
            sampn = min(sampn, len(nextboards))
            for nextboard in sample(nextboards, sampn):
                sampvalue, nextact = ExpectiMax(nextboard, level=level-1)
                value_col.append(sampvalue)
            nextvalue = sum(value_col) / sampn
        else:
            nextvalue = 0 # evaluation function!
        curvalue = nextvalue + reward - (finished) * 1000 # punishment for death
        if curvalue > bestVal: bestVal, bestAct = curvalue, act
    return bestAct, bestVal

def gameSimul(policy, policyArgs={}, initboard=None, initscore=0, printfreq=50):
    board = getInitState() if initboard is None else initboard
    score = initscore
    actseq = []
    while True:
        # act, bestVal, = RndMax(board, 3)
        act, bestVal, = policy(board, **policyArgs)
        # act = choice(actions)
        actseq.append(act)
        board, reward, finished = getSuccessor(board, action=act, show=False)
        score += reward
        if printfreq != 0 and (len(actseq) % printfreq == 0):
            print("Step %d score %d"%(len(actseq), score))
            print(board)
        if finished:
            print("Game Over, step %d score %d"%(len(actseq), score))
            break
    return actseq, score
#%%

if __name__ == "__main__":
    # Let AI play this!
    board = getInitState()
    score = 0
    actseq = []
    while True:
        # act, bestVal, = RndMax(board, 3)
        act, bestVal, = ExpectiMax(board, 3, sampn=3)
        # act = choice(actions)
        actseq.append(act)
        board, reward, finished = getSuccessor(board, action=act, show=False)
        score += reward
        if len(actseq) % 50 == 0:
            print("Step %d score %d"%(len(actseq), score))
            print(board)
        if finished:
            print("Game Over, step %d score %d"%(len(actseq), score))
            break

# board = 0*np.ones((4, 4), dtype=np.int) # [[None for i in range(4)] for i in range(4)]
# blocklist = [(1, (1, 1)), (1, (2, 1)), (1, (1, 2))]
# for num, (x, y) in blocklist:
#     board[x, y] = num
#
# seq = [num for num in board[:, 1] if num!=0]
#
# board, reward_all = fallBoard(board, dir=LEFT);print(board)
# board, reward_all = fallBoard(board, dir=DOWN);print(board)
# board, reward_all = fallBoard(board, dir=UP);print(board)
# board, reward_all = fallBoard(board, dir=RIGHT);print(board)

# board = NULLVAL*np.ones((4, 4), dtype=np.int) # [[None for i in range(4)] for i in range(4)]
# blocklist = [(1, (1, 1)), (1, (2, 1)), (1, (1, 2))]
# for num, (x, y) in blocklist:
#     board[x, y] = num
#
# score = 0
# board, reward, finished = getSuccessor(board, action=LEFT); score += reward