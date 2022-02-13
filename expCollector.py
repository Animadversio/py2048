from main import ExpectiMax, RndMax
from main import getSuccessor, getInitState
import numpy as np
def traj_sampler(policy, initboard=None, initscore=0, policyArgs={}, printfreq=50):
    """ Trajectory sampler with a given policy.
    policy: policy function
    policyArgs: dict of arguments to policy function.
    initboard: initial board state.
    printfreq: positive integer, print frequency; 0 for not print until finish; -1 for total silence.

    return :
        stateseq: list of array representing boards, length T+1; starts from s_0
        actseq: list of scalar, action indices, length T; starts from a_1
        rewardseq: list of scalar, length T; starts from r_1
        sum_reward: sum(rewardseq) total reward gain in this trajectory, without decay.
    """
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
        if printfreq != 0 and printfreq != -1 and (len(actseq) % printfreq == 0):
            print("Step %d score %d" % (len(actseq), score))
            print(board)
        if finished:
            if printfreq != -1:
                print("Game Over, step %d score %d" % (len(actseq), score))
            break
    return stateseq, actseq, rewardseq, sum(rewardseq)


episode_buffer = {}
def episodeLoader(triali, episode_buffer=episode_buffer, savetensor=False):
    if triali not in episode_buffer:
        data = np.load("exp_data\\traj%03d.npz"%triali)
        actseq = data['actseq']  # (T, )
        rewardseq = data['rewardseq']  # (T, )
        stateseq = data['stateseq']  # (T+1, 4, 4)
        score_tot = data['score']
        if savetensor:
            episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                                     torch.tensor(stateseq), score_tot
            return torch.tensor(actseq), torch.tensor(rewardseq), \
                                     torch.tensor(stateseq), score_tot
        else:
            episode_buffer[triali] = actseq, rewardseq, stateseq, score_tot
            return actseq, rewardseq, stateseq, score_tot
    else:
        actseq, rewardseq, stateseq, score_tot = episode_buffer[triali]
        return actseq, rewardseq, stateseq, score_tot


def episodeSaver(triali, actseq, rewardseq, stateseq, score_tot, episode_buffer=episode_buffer, savetensor=False):
    if savetensor:
        episode_buffer[triali] = torch.tensor(actseq), torch.tensor(rewardseq), \
                                 torch.tensor(stateseq), score_tot
    else:
        episode_buffer[triali] = actseq, rewardseq, stateseq, score_tot


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