from main import ExpectiMax, RndMax
from main import getSuccessor, getInitState
import numpy as np
def traj_sampler(policy, initboard=None, initscore=0, policyArgs={}, printfreq=50):
    # act, bestVal, = RndMax(board, 3)
    # act, bestVal, = ExpectiMax(board, 2, sampn=4)
    # act = choice(actions)
    board = getInitState() if initboard is None else initboard
    score = 0
    actseq = []
    stateseq = []
    rewardseq = []
    stateseq.append(board)  # initial state
    while True:
        act, bestVal = policy(board, **policyArgs)
        actseq.append(act)
        board, reward, finished = getSuccessor(board, action=act, show=False)
        stateseq.append(board)
        rewardseq.append(reward)
        score += reward
        if printfreq != 0 and (len(actseq) % printfreq == 0):
            print("Step %d score %d" % (len(actseq), score))
            print(board)
        if finished:
            print("Game Over, step %d score %d" % (len(actseq), score))
            break
    return stateseq, actseq, rewardseq, sum(rewardseq)
#%%
if __name__=="__main__":
    from time import time
    from tqdm import tqdm
    final_scores = []
    for triali in tqdm(range(1000)):
        board = getInitState()
        score = 0
        actseq = []
        stateseq = []
        rewardseq = []
        stateseq.append(board) # initial state
        while True:
            # act, bestVal, = RndMax(board, 3)
            act, bestVal, = ExpectiMax(board, 2, sampn=4)
            # act = choice(actions)
            actseq.append(act)
            board, reward, finished = getSuccessor(board, action=act, show=False)
            stateseq.append(board)
            rewardseq.append(reward)
            score += reward
            if len(actseq) % 50==0:
                print("Step %d score %d"%(len(actseq), score))
                print(board)
            if finished:
                print("Game Over, step %d score %d"%(len(actseq), score))
                break
        np.savez("exp_data\\traj%03d.npz"%triali, stateseq=stateseq, actseq=actseq, rewardseq=rewardseq, score=score)
        final_scores.append(score)

    np.savez("exp_data\\expectimax_scores.npz", scores=final_scores)
    #%%
    import matplotlib.pylab as plt
    plt.figure()
    plt.hist(final_scores,50);plt.xlabel("final score");plt.title("Multi-Sample Expectimax")
    plt.savefig("expectimax.png")  # depth 2, sample N 4
    plt.show()