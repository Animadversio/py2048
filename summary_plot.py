
import numpy as np
import matplotlib.pylab as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
plt.figure()
data3 = np.load(r"E:\Github_Projects\2048\exp_data\DQN_3dCNN_18M_scores.npz",)
scorevec = data3["eps_rew"]
plt.hist(scorevec, bins=65, label="DQN", alpha=0.4)
plt.axvline(x=scorevec.mean(), label=f"DQN: {scorevec.mean():.0f}+-{scorevec.std():.0f}", ls="-", color="red")
data2 = np.load(r"E:\Github_Projects\2048\exp_data\PPO_3dCNN_28M_scores.npz",)
scorevec = data2["eps_rew"]
plt.hist(scorevec, bins=65, label="PPO", alpha=0.4)
plt.axvline(x=scorevec.mean(), label=f"PPO: {scorevec.mean():.0f}+-{scorevec.std():.0f}", ls="-.")
data = np.load(r"E:\Github_Projects\2048\exp_data\expectimax_scores.npz")
scorevec = data["scores"]
plt.hist(scorevec, bins=65, label="ExpectiMax", alpha=0.4, )
plt.axvline(x=scorevec.mean(), label=f"ExpectiMax: {scorevec.mean():.0f}+-{scorevec.std():.0f}", ls=":", color="k")
plt.title(f"Comparison of PPO, DQN, and Expectimax scores (N=1000)")
plt.xlabel("Episode Reward")
plt.legend()
plt.savefig("score_comparison_DQN_PPO_ExpectiMax.png")
plt.savefig("score_comparison_DQN_PPO_ExpectiMax.pdf")
plt.show()
