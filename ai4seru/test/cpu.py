import os

# 获取逻辑CPU数量（包含超线程）
logical_cores = os.cpu_count()
print(f"逻辑CPU核心数: {logical_cores}")