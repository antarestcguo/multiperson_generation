环境配置
pip install -r flux_requirements.txt

算法简要说明
1. 基础算法：使用flux生成底图，face区域inpainting换脸
2. 改进：
   - clip匹配生成任务和character
   - face区域膨胀高斯
   - 分别crop face区域逐个换脸，再resize回原图
   - prompt优化

