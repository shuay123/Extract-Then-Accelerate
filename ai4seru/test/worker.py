import numpy as np
import pandas as pd


def calculate_cosine_similarity(vector1, vector2):
    """计算两个向量的余弦相似度"""
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cosine_sim = np.dot(vector1, vector2) / (norm1 * norm2)
    return cosine_sim


def analyze_worker_similarities():
    """分析所有工人之间的余弦相似度"""

    # 工人能力数据
    worker_abilities = {
        1: [2.44, 1, 1.98, 1.985714, 0.2],
        2: [1.52901, 1.52901, 1.52901, 1.52901, 1.52901],
        3: [1.385714, 0.3, 0.875, 1.885714, 1.15],
        4: [1, 0.45, 1.6, 1.46, 0.3],
        5: [2.4, 1.1, 2.433333, 1.14, 1.533333],
        6: [2.255882, 0.2, 1.9125, 1.762069, 0.92],
        7: [1.050238, 1.050238, 1.050238, 1.050238, 1.050238],
        8: [1.984211, 1.6, 1.3, 1.573684, 0.9],
        9: [0.4, 1.7, 1.3, 2.033333, 1.241689],
        10: [1.3981, 1.3981, 1.3981, 1.3981, 1.3981],
        11: [1.31, 0.3, 1, 1.155556, 0.7],
        12: [1.047253, 1.047253, 1.047253, 1.047253, 1.047253],
        13: [2.036364, 1.5, 1.620024, 1.516667, 0.2],
        14: [1.538235, 0.8, 1.183333, 1.7, 0.45],
        15: [2.003333, 0.266667, 0.65, 1.65, 0.25]
    }

    # 转换为numpy数组
    worker_vectors = {}
    for worker_id, abilities in worker_abilities.items():
        worker_vectors[worker_id] = np.array(abilities)

    # 计算所有工人对之间的余弦相似度
    num_workers = len(worker_vectors)
    worker_ids = list(worker_vectors.keys())

    print("=== 工人能力向量 ===")
    for worker_id in worker_ids:
        vector = worker_vectors[worker_id]
        variance = np.var(vector)
        mean_ability = np.mean(vector)
        print(f"工人{worker_id:2d}: {vector} (均值={mean_ability:.3f}, 方差={variance:.3f})")

    print("\n=== 工人之间余弦相似度矩阵 ===")

    # 创建相似度矩阵
    similarity_matrix = np.zeros((num_workers, num_workers))

    # 打印表头
    print("工人\\工人", end="")
    for worker_id in worker_ids:
        print(f"{worker_id:8d}", end="")
    print()

    # 计算并打印相似度矩阵
    for i, worker_i in enumerate(worker_ids):
        print(f"工人{worker_i:2d}", end="  ")
        for j, worker_j in enumerate(worker_ids):
            if i == j:
                similarity = 1.0000
            else:
                similarity = calculate_cosine_similarity(
                    worker_vectors[worker_i],
                    worker_vectors[worker_j]
                )
            similarity_matrix[i][j] = similarity
            print(f"{similarity:8.4f}", end="")
        print()

    print("\n=== 最相似的工人对 ===")
    similar_pairs = []
    for i, worker_i in enumerate(worker_ids):
        for j, worker_j in enumerate(worker_ids):
            if i < j:  # 避免重复和自己与自己比较
                similarity = similarity_matrix[i][j]
                similar_pairs.append((worker_i, worker_j, similarity))

    # 按相似度排序
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    print("最相似的前10对工人:")
    for i, (worker_i, worker_j, similarity) in enumerate(similar_pairs[:10]):
        print(f"{i + 1:2d}. 工人{worker_i} - 工人{worker_j}: {similarity:.4f}")

    print("\n=== 最不相似的工人对 ===")
    print("最不相似的前10对工人:")
    for i, (worker_i, worker_j, similarity) in enumerate(similar_pairs[-10:]):
        print(f"{i + 1:2d}. 工人{worker_i} - 工人{worker_j}: {similarity:.4f}")

    print("\n=== 特殊工人分析 ===")

    # 找出能力最均匀的工人（方差最小）
    uniform_workers = []
    for worker_id in worker_ids:
        vector = worker_vectors[worker_id]
        variance = np.var(vector)
        uniform_workers.append((worker_id, variance))

    uniform_workers.sort(key=lambda x: x[1])
    print("能力最均匀的工人（方差最小）:")
    for i, (worker_id, variance) in enumerate(uniform_workers[:5]):
        print(f"{i + 1}. 工人{worker_id}: 方差={variance:.6f}")

    # 找出能力最多样的工人（方差最大）
    print("\n能力最多样的工人（方差最大）:")
    for i, (worker_id, variance) in enumerate(uniform_workers[-5:]):
        print(f"{i + 1}. 工人{worker_id}: 方差={variance:.6f}")

    # 分析相似度统计
    all_similarities = []
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            all_similarities.append(similarity_matrix[i][j])

    print(f"\n=== 相似度统计 ===")
    print(f"平均相似度: {np.mean(all_similarities):.4f}")
    print(f"相似度标准差: {np.std(all_similarities):.4f}")
    print(f"最高相似度: {np.max(all_similarities):.4f}")
    print(f"最低相似度: {np.min(all_similarities):.4f}")

    # 保存到CSV文件
    df = pd.DataFrame(similarity_matrix,
                      index=[f"工人{i}" for i in worker_ids],
                      columns=[f"工人{i}" for i in worker_ids])
    df.to_csv("worker_cosine_similarities.csv", encoding='utf-8-sig')
    print(f"\n相似度矩阵已保存到 worker_cosine_similarities.csv")

    return similarity_matrix, worker_ids


if __name__ == "__main__":
    similarity_matrix, worker_ids = analyze_worker_similarities()