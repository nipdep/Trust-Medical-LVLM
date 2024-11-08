from src.evaluators.metrics import _supported_metrics
import numpy as np

# 实际标签
y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# 模型预测结果
y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 0, 0])
# 群体划分 (0 表示女性, 1 表示男性)
group = np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0])

# 调用函数前需要确保传入正确的布尔掩码
error_ratio_f, error_ratio_m = _supported_metrics["treatment_equality"](y_true, y_pred, group)

print(f"Error Ratio for Group 0 (女性): {error_ratio_f}")
print(f"Error Ratio for Group 1 (男性): {error_ratio_m}")