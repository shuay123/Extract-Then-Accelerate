from sklearn.metrics import adjusted_rand_score

# 真正划分：A,B 属于簇 0；C,D 属于簇 1
labels_true = [1, 1, 1, 0]

# 预测划分：A,C 属于簇 0；B,D 属于簇 1
labels_pred = [0, 1, 2 ,4]

ari = adjusted_rand_score(labels_true, labels_pred)
print(f"Adjusted Rand Index: {ari:.4f}")
