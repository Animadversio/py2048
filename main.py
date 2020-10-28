

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

#%%
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

def gameSimul(policy, policyArgs={}, initboard=None, initscore=0):
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
        if len(actseq) % 50==0:
            print("Step %d score %d"%(len(actseq), score))
            print(board)
        if finished:
            print("Game Over, step %d score %d"%(len(actseq), score))
            break
    return actseq, score
#%%
import pygame as pg
pg.display.set_caption("SLG board")
SCREEN = pg.display.set_mode((400, 500)) # get screen
screen = pg.display.get_surface()
# Setup Font
pg.font.init()
font = pg.font.SysFont("microsoftsansserif", 20, bold=False)
board_WIDTH, board_HEIGHT = 400, 500
TileW = 100
Margin = 5
NUMCOLOR = {NULLVAL: (200, 200, 200), 1: (190, 190, 200), 2: (180, 180, 200), 4: (180, 180, 210), 8: (170, 170, 210),
            16: (165, 165, 220), 32: (155, 155, 220), 64: (145, 145, 225), 128: (135, 135, 225), 256: (125, 125, 230),
            512: (110, 110, 240), 1024: (95, 95, 240), 2048: (80, 80, 250)}

def drawBoard(board, score):
    pg.draw.rect(screen, (240, 240, 240), pg.Rect(0, 0, board_WIDTH, board_HEIGHT), 0)
    for i in range(dimen):
        for j in range(dimen):
            clr = NUMCOLOR[board[i, j]]
            pg.draw.rect(screen, clr,
                         pg.Rect(i * TileW + Margin, j * TileW + Margin, TileW - 2 * Margin, TileW - 2 * Margin))
            if board[i, j] != NULLVAL:
                img = font.render('%d' % board[i, j], True, (0,0,0))
                screen.blit(img, ((i + 1/2) * TileW - 28, (j + 1/2) * TileW - 18))
    scoreimg = font.render('%d' % score, True, (100, 40, 40))
    screen.blit(scoreimg, (200, 450))

def GUI_loop(board=None, score=0):
    if board is None:
        board = getInitState()
        score = 0
    exitFlag = False
    actseq = []
    while not exitFlag:
        act = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exitFlag = True
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP: act = LEFT
                if event.key == pg.K_DOWN: act = RIGHT
                if event.key == pg.K_LEFT: act = UP
                if event.key == pg.K_RIGHT: act = DOWN
        if act is not None:
            actseq.append(act)
            board, reward, finished = getSuccessor(board, action=act, show=False)
            score += reward
            if finished:
                exitFlag = True
        drawBoard(board, score)
        pg.display.update()

if __name__ == "__main__":
    # Try GUI version
    # GUI_loop(board=None, score=0)
    # Let AI play this!
    board = getInitState()
    score = 0
    actseq = []
    while True:
        # act, bestVal, = RndMax(board, 3)
        act, bestVal, = ExpectiMax(board, 2, sampn=4)
        # act = choice(actions)
        actseq.append(act)
        board, reward, finished = getSuccessor(board, action=act, show=False)
        score += reward
        if len(actseq) % 50==0:
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