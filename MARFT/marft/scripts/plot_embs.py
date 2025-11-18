import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np

# ---------------------
# 1. Extract REDTEAMER actions
# ---------------------
def extract_redteam_actions(path):
    actions = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("REDTEAMER action: "):
                action = line[len("REDTEAMER action: "):].strip()
                actions.append(action)
    return actions

file1 = "/home/a38das/MARFT/marft/scripts/redteam_sql_moredata.log"
# file2 = "/home/a38das/MARFT/marft/scripts/marft_v5_3_reward1.log"
file2 = "/home/a38das/MARFT/marft/scripts/marft_v2.log"

actions1 = extract_redteam_actions(file1)
actions2 = extract_redteam_actions(file2)

print(f"Set 1: {len(actions1)} actions")
print(f"Set 2: {len(actions2)} actions")

# ---------------------
# 2. Compute embeddings
# ---------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

emb1 = model.encode(actions1)
emb2 = model.encode(actions2)

# ---------------------
# 3. Reduce to 2D
# (Use PCA, or replace with TSNE if preferred)
# ---------------------
pca = PCA(n_components=2)
emb_all = pca.fit_transform(np.vstack([emb1, emb2]))

N1 = len(emb1)
emb1_2d = emb_all[:N1]
emb2_2d = emb_all[N1:]

# ---------------------
# 4. Plot both sets
# ---------------------
plt.figure(figsize=(10, 8))

# Plot set 1
plt.scatter(emb1_2d[:,0], emb1_2d[:,1], label="Set 1", alpha=0.7)

# Plot set 2
plt.scatter(emb2_2d[:,0], emb2_2d[:,1], label="Set 2", alpha=0.7)

plt.title("Text Embedding Visualization for Two REDTEAMER Action Sets")
plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
# plt.show()
plt.savefig('embeddings_plot2.png')
