import numpy as np
import random
from typing import List
from utils.excel_utils import ExcelDataLoader


class WorkerCollaborationAnalyzer:
    """工人协作能力分析器"""

    def __init__(self, excel_loader: ExcelDataLoader, num_of_workers: int):
        """
        初始化分析器

        Args:
            excel_loader: Excel数据加载器
            num_of_workers: 工人总数
        """
        self.excel_loader = excel_loader
        self.num_of_workers = num_of_workers

    def calculate_formation_similarity(self, formation_code: List[int]) -> float:
        """
        计算SeruFormation的工人协作能力（相似程度）

        Args:
            formation_code: formation代码

        Returns:
            工人能力相似程度的平均值
        """
        # 根据formation_code解析seru结构
        seru_worker_groups = self._parse_formation_code(formation_code)

        seru_similarities = []

        for workers_in_seru in seru_worker_groups:
            if len(workers_in_seru) <= 1:
                # 单个工人的seru单元，余弦相似度设为1（完全相似）
                seru_similarities.append(1.0)
                continue

            # 获取工人能力向量
            worker_abilities = []
            for worker_id in workers_in_seru:
                ability_vector = self._get_worker_ability_vector(worker_id)
                worker_abilities.append(ability_vector)

            # 计算工人之间两两余弦相似度
            cosine_similarities = []
            for i in range(len(worker_abilities)):
                for j in range(i + 1, len(worker_abilities)):
                    cosine_sim = self._calculate_cosine_similarity(
                        worker_abilities[i], worker_abilities[j]
                    )
                    cosine_similarities.append(cosine_sim)

            # 取该seru单元内工人相似度的均值
            if cosine_similarities:
                seru_avg_similarity = np.mean(cosine_similarities)
            else:
                seru_avg_similarity = 1.0  # 没有工人对的情况

            seru_similarities.append(seru_avg_similarity)

        # 对所有seru单元取均值
        formation_similarity = np.mean(seru_similarities)
        return formation_similarity

    def generate_random_worker_assignment(self, base_formation_code: List[int]) -> List[int]:
        """
        固定formation结构，随机打乱工人分配

        Args:
            base_formation_code: 基础的formation代码

        Returns:
            新的formation代码，结构相同但工人分配随机化
        """
        # 分离工人和分隔符
        workers = [code for code in base_formation_code if code <= self.num_of_workers]

        # 随机打乱工人顺序
        random.shuffle(workers)

        # 重新组装formation_code
        new_formation_code = []
        worker_index = 0

        for code in base_formation_code:
            if code <= self.num_of_workers:
                # 用打乱后的工人替换
                new_formation_code.append(workers[worker_index])
                worker_index += 1
            else:
                # 保持分隔符不变
                new_formation_code.append(code)

        return new_formation_code

    def _parse_formation_code(self, formation_code: List[int]) -> List[List[int]]:
        """
        解析formation_code为seru工人分组

        Args:
            formation_code: formation代码

        Returns:
            每个seru的工人列表
        """
        seru_groups = []
        current_group = []

        for code in formation_code:
            if code <= self.num_of_workers:
                current_group.append(code)
            else:
                if current_group:
                    seru_groups.append(current_group)
                current_group = []

        if current_group:
            seru_groups.append(current_group)

        return seru_groups

    def _get_worker_ability_vector(self, worker_id: int) -> np.ndarray:
        """
        获取工人的能力向量（对所有产品类型的熟练程度）

        Args:
            worker_id: 工人ID

        Returns:
            工人能力向量
        """
        if worker_id not in self.excel_loader.worker_to_product_dict:
            raise ValueError(f"Worker ID {worker_id} not found in worker_to_product_dict")

        worker_data = self.excel_loader.worker_to_product_dict[worker_id]

        # 获取所有产品类型的能力值
        ability_values = []
        for product_type in range(1, 6):  # 假设有5种产品类型（1-5）
            if product_type in worker_data:
                ability_values.append(worker_data[product_type])
            else:
                ability_values.append(0.0)  # 如果没有该产品类型的能力，设为0

        return np.array(ability_values)

    def _calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vector1: 向量1
            vector2: 向量2

        Returns:
            余弦相似度值 (-1到1之间)
        """
        # 避免除零错误
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = np.dot(vector1, vector2) / (norm1 * norm2)
        return cosine_sim