import os
import sys

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root)

# 导入 app.config 会加载 .env 到 os.environ。
# Chains/*.py 在 import 期读取 OPENAI_API_KEY，因此这行必须早于它们的导入。
import app.config  # noqa: F401,E402
