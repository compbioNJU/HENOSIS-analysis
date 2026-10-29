"""Write a concise report for the two glial regional comparisons."""
import pandas as pd
from settings import RESULTS
T=RESULTS/'tables'
stats=pd.read_csv(T/'figure_l_regional_statistics.csv')
heat=pd.read_csv(T/'figure_l_regional_heatmap.csv')
lines=['# Figure 4 k–l：小鼠胶质细胞的脑区年龄变化','',
       '细胞比例使用各谱系、脑区的年轻参考阈值；图中每点代表一只小鼠，连接线表示年龄组均值。','',
       '区域对比为同一只小鼠的 CC/ACO 减 CTX，再比较老年组与年轻组。q 为两项区域对比的探索性双侧全流程置换 BH 校正。','']
for r in stats.itertuples():
 lines.append(f'- {r.cell_type}: 区域差值的老年减年轻变化 {r.full_cell_older_minus_young_regional_contrast_pp:+.1f} percentage points；q = {r.q_two_exploratory:.4f}。')
lines.extend(['','各脑区的老年减年轻变化为描述性估计；完整个体数据和检验数值位于 tables。',''])
print('\n'.join(lines))
