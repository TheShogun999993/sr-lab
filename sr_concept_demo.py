"""Quick concept demo of the Successor Representation (Stachenfeld et al. 2017).

Produces the figures used to explain the project concept. Pure NumPy + matplotlib.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "."


# ---------------------------------------------------------------- engine ----
def grid_env(h, w, walls=()):
    """4-neighbour grid. Returns (free cell list, index map, adjacency A)."""
    walls = set(walls)
    cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in walls]
    idx = {cell: i for i, cell in enumerate(cells)}
    A = np.zeros((len(cells), len(cells)))
    for (r, c), i in idx.items():
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                A[i, j] = 1
    return cells, idx, A


def random_walk_T(A):
    return A / A.sum(1, keepdims=True)


def successor(T, gamma):
    """Eq. 7: M = (I - gamma T)^-1."""
    return np.linalg.inv(np.eye(len(T)) - gamma * T)


def to_image(vec, cells, h, w):
    img = np.full((h, w), np.nan)
    for v, (r, c) in zip(vec, cells):
        img[r, c] = v
    return img


def eigvecs_random_walk(A):
    """Right eigenvectors of T = D^-1 A (shared by M), sorted low->high frequency."""
    d = A.sum(1)
    Dm = np.diag(d ** -0.5)
    lam, U = np.linalg.eigh(Dm @ A @ Dm)
    order = np.argsort(-lam)
    return lam[order], (Dm @ U)[:, order]


def show(ax, img, title, cmap="viridis"):
    m = np.ma.masked_invalid(img)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#333333")  # walls
    ax.imshow(m, cmap=cm, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


GAMMA = 0.95

# ------------------------------------------- Fig A: wall bends place fields --
H, W = 12, 12
wall = [(r, 6) for r in range(0, 9)]           # wall from top, gap at bottom
target = (3, 5)                                 # cell just left of the wall
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
# Euclidean Gaussian (traditional view)
cells, idx, A = grid_env(H, W, wall)
g = np.array([np.exp(-((r - target[0]) ** 2 + (c - target[1]) ** 2) / 8) for r, c in cells])
show(axs[0], to_image(g, cells, H, W), "Traditional 'GPS' place field\n(ignores the wall)")
# SR, open room
c0, i0, A0 = grid_env(H, W)
M0 = successor(random_walk_T(A0), GAMMA)
show(axs[1], to_image(M0[:, i0[target]], c0, H, W), "SR place field, open room\nM(:, s')")
# SR with wall
M = successor(random_walk_T(A), GAMMA)
show(axs[2], to_image(M[:, idx[target]], cells, H, W), "SR place field, wall added\n(bends around the wall)")
for ax in axs:
    ax.plot(target[1], target[0], "r+", ms=12, mew=2)
plt.suptitle("A. Place cell = 'how soon will I get to s' from here?'", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/figA_wall.png", dpi=110); plt.close()

# ----------------------------------------- Fig B: backward skew on a track --
N, s_star = 41, 25
def track_T(p_right):
    T = np.zeros((N, N))
    for i in range(N):
        r, l = min(i + 1, N - 1), max(i - 1, 0)
        T[i, r] += p_right; T[i, l] += 1 - p_right
    return T
fig, ax = plt.subplots(figsize=(8, 3.4))
for p, lab, col in ((0.5, "no bias (50/50)", "tab:red"), (0.9, "runs right 90% of the time", "tab:blue")):
    f = successor(track_T(p), 0.9)[:, s_star]
    ax.plot(f / f.max(), label=lab, color=col, lw=2)
ax.axvline(s_star, color="k", ls="--", lw=1); ax.text(s_star + 0.5, 1.02, "cell's preferred spot", fontsize=9)
ax.annotate("", xy=(30, 0.15), xytext=(18, 0.15), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(19, 0.2, "direction of travel", fontsize=9)
ax.set_xlabel("position on track"); ax.set_ylabel("firing (normalised)")
ax.set_title("B. With a habit of running right, the field skews BACKWARDS (Mehta 2000 / paper Fig. 3b)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9); plt.tight_layout(); plt.savefig(f"{OUT}/figB_skew.png", dpi=110); plt.close()

# -------------------------------------- Fig C: instant reward revaluation ---
fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
show(axs[0], to_image(M[idx[(10, 2)]], cells, H, W), "SR row M(s, :) at the red dot\n(learned once)", "magma")
axs[0].plot(2, 10, "r+", ms=12, mew=2)
for ax, goal, name in ((axs[1], (1, 10), "Reward top-right"), (axs[2], (1, 1), "Reward moved top-left")):
    R = np.zeros(len(cells)); R[idx[goal]] = 1
    V = M @ R                                     # Eq. 2: V = M R, no relearning
    show(ax, to_image(V, cells, H, W), f"{name}\nV = M R  (one multiply)", "magma")
    ax.plot(goal[1], goal[0], "c*", ms=14)
plt.suptitle("C. Same map, new goal: value recomputed instantly (paper Fig. 1)", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/figC_reward.png", dpi=110); plt.close()

# ---------------------- Fig D: grid-cell-like eigenvectors + doorway subgoal --
H2, W2 = 11, 23
wall2 = [(r, 11) for r in range(H2) if r != 5]  # two rooms, door at row 5
cells2, idx2, A2 = grid_env(H2, W2, wall2)
lam, U = eigvecs_random_walk(A2)
fig, axs = plt.subplots(2, 3, figsize=(12, 5.2))
v1 = U[:, 1]
show(axs[0, 0], to_image(v1, cells2, H2, W2), "Eigenvector #1: splits the rooms", "coolwarm")
sign = v1 > 0
sub = [cells2[i] for i in range(len(cells2))
       if any(A2[i, j] and sign[i] != sign[j] for j in range(len(cells2)))]
axs[0, 0].plot([c for r, c in sub], [r for r, c in sub], "y*", ms=14)
axs[0, 0].set_title("Eigenvector #1 splits the rooms\n* = discovered subgoal (doorway)", fontsize=10)
for ax, k in zip(list(axs.flat)[1:], (4, 7, 12, 18, 27)):
    show(ax, to_image(np.maximum(U[:, k], 0), cells2, H2, W2), f"Eigenvector #{k} (thresholded at 0)", "inferno")
plt.suptitle("D. 'Grid cells' = eigenvectors of the SR; the biggest one finds the doorway (paper Figs 6-8, Supp. 13)",
             fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/figD_eigen.png", dpi=110); plt.close()

# ------------------------------------------------------------ sanity check --
print("row sums ~ 1/(1-gamma):", M0.sum(1)[:3].round(3), "expected", round(1 / (1 - GAMMA), 3))
f50 = successor(track_T(0.5), 0.9)[:, s_star]; f90 = successor(track_T(0.9), 0.9)[:, s_star]
print("track mass behind vs ahead  50/50:", f50[:s_star].sum().round(2), f50[s_star + 1:].sum().round(2),
      " 90/10:", f90[:s_star].sum().round(2), f90[s_star + 1:].sum().round(2))
print("subgoal cells:", sub)
