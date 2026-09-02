import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import List

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class SimpleCorrelationAnalyzer:
    """
    简化版相关性分析和线性回归工具类
    """

    def __init__(self, x_data: List[float], y_data: List[float],
                 x_name: str = "X", y_name: str = "Y"):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.x_name = x_name
        self.y_name = y_name

    def analyze(self):
        """
        进行相关性分析和线性回归
        """
        # 相关性分析
        correlation, p_value = stats.pearsonr(self.x_data, self.y_data)

        # 线性回归
        X = self.x_data.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, self.y_data)
        y_pred = model.predict(X)
        r2 = r2_score(self.y_data, y_pred)

        # 打印结果
        print("=" * 50)
        print("相关性分析和线性回归结果")
        print("=" * 50)
        print(f"Pearson相关系数: {correlation:.4f}")
        if p_value < 0.001:
            print(f"显著性 (p值): {p_value:.2e} (高度显著)")
        else:
            print(f"显著性 (p值): {p_value:.4f} ({'显著' if p_value < 0.05 else '不显著'})")
        print(f"回归方程: {self.y_name} = {model.coef_[0]:.2f} * {self.x_name} + {model.intercept_:.2f}")
        print(f"R²: {r2:.4f}")
        print("=" * 50)

        # 绘图
        plt.figure(figsize=(10, 6))

        # 散点图和回归线
        plt.subplot(1, 2, 1)
        plt.scatter(self.x_data, self.y_data, alpha=0.6, color='blue')
        plt.plot(self.x_data, y_pred, color='red', linewidth=2, label='回归线')
        plt.xlabel(self.x_name)
        plt.ylabel(self.y_name)
        plt.title(f'散点图和回归线\nR² = {r2:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 残差图
        plt.subplot(1, 2, 2)
        residuals = self.y_data - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return correlation, p_value, r2, model


# 使用示例
if __name__ == "__main__":
    # Input data
    similarity_data = [0.919, 0.924, 0.927, 0.928, 0.931, 0.932, 0.933, 0.934, 0.935,
                       0.936, 0.937, 0.938, 0.942, 0.943, 0.944, 0.945, 0.947, 0.947,
                       0.949, 0.95, 0.95, 0.95, 0.951, 0.952, 0.952, 0.952, 0.953,
                       0.953, 0.953, 0.954, 0.954, 0.955, 0.956, 0.957, 0.958, 0.96,
                       0.961, 0.961, 0.961, 0.961, 0.962, 0.962, 0.964, 0.965, 0.965,
                       0.967, 0.968, 0.973, 0.975, 0.977]

    tardiness_data = [105176, 104544, 94196, 77771, 101606, 106084, 34055, 43919, 111425,
                      25406, 23475, 73727, 50676, 2806, 26451, 30132, 13296, 133686,
                      59815, 14038, 36434, 0, 8197, 11221, 70650, 3519, 58956, 0,
                      26750, 35371, 20136, 15431, 77372, 19549, 8520, 55163, 15750,
                      0, 6619, 8992, 4078, 0, 4973, 0, 0, 0, 11645, 0, 7653, 0]

    # 创建分析器并分析
    analyzer = SimpleCorrelationAnalyzer(
        x_data=similarity_data,
        y_data=tardiness_data,
        x_name="工人协作相似度",
        y_name="总延误时间"
    )

    correlation, p_value, r2, model = analyzer.analyze()
