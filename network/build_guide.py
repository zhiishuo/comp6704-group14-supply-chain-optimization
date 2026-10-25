"""Create the Chinese reading and presentation guide from final statistics."""
from pathlib import Path
import json,html
ROOT=Path(__file__).absolute().parent.parent
s=json.loads((ROOT/'network/results/summary.json').read_text(encoding='utf-8'))
large=[g for g in s['groups'] if g['variables']==32400]
rows=''.join(f'<tr><td>{html.escape(g["method"])}</td><td>{g["median"]/1000:.3f} 秒</td><td>{g["success"]}/{g["n"]}</td><td>{g["max_residual"]:.2e}</td></tr>' for g in large)
doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>供应链优化项目｜中文阅读说明</title>
<style>body{margin:0;background:#eef2f5;color:#233645;font:16px/1.9 "Microsoft YaHei",sans-serif}main{max-width:1000px;margin:35px auto;background:white;padding:50px 60px;border-top:7px solid #087f8c}h1{font-size:31px;line-height:1.5}h2{font-size:23px;border-bottom:1px solid #d9e2e8;margin-top:38px;padding-bottom:8px}h3{font-size:18px}p{margin:12px 0}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px 12px;border-bottom:1px solid #dae2e8;text-align:left}th{background:#eff6f7}img{width:100%;height:auto;margin:18px 0}.meta{color:#607583}.note{border-left:4px solid #087f8c;background:#f2f8f8;padding:14px 20px}a{color:#2456a2}.formula{background:#f2f5f7;padding:12px 18px;font-family:"Cambria Math",serif}@media print{body{background:white}main{margin:0;padding:20px;border:0}h2{break-after:avoid}table,img{break-inside:avoid}}</style><main>
<p class="meta">COMP6704 · Linear Programming · 2026-10-24</p>
<h1>带碳排放约束的两级供应链运输优化<br>经典算法与现代线性规划方法比较</h1>
<p><a href="README.md">代码与复现说明</a></p>
<div class="note">本仓库提供 COMP6704 Group 14 的算法源码、模拟实例、归档实验与复现说明。团队已报告复现代码。所有业务数据均为明确构造的模拟数据；课程报告和演示文稿单独提供。</div>
<h2>1. 现在研究的是什么问题</h2>
<p>货物由工厂经过仓库运往配送中心。我们决定每条线路运输多少货物，使运输与仓库处理总成本最低，同时满足运营能力和碳排放预算。工厂和仓库都已经存在，因此不需要使用“建或不建”的 0–1 变量，整个项目仍然是连续线性规划。</p>
<p>最大算例包含 <strong>90 个工厂、120 个仓库、180 个配送中心、32,400 个运输变量</strong>，共有 511 条等式或上界约束，另有各变量的线路容量上下界。该规模适合课程实验与算法比较，但不宣称是工业级超大规模记录。</p>
<img src="network/figures/shipment_network.png" alt="优化后的两级供应链示意图">
<h2>2. 六类约束怎样分别解释</h2>
<table><tr><th>约束</th><th>数学含义</th><th>解释要点</th></tr>
<tr><td>工厂供给</td><td>每个工厂发出的总量 ≤ 可供给量</td><td>允许不用满产能；不能强行设成等式。</td></tr>
<tr><td>仓库流量平衡</td><td>每个仓库流入量 = 流出量</td><td>保证两级运输连接起来；本模型没有跨期库存。</td></tr>
<tr><td>需求满足</td><td>每个配送中心收货量 = 需求量</td><td>不允许缺货，也不多送。</td></tr>
<tr><td>仓库吞吐量</td><td>仓库总入库量 ≤ 处理能力</td><td>限制仓库整体能力；流入流出只计一次。</td></tr>
<tr><td>线路容量</td><td>0 ≤ 每条线路运输量 ≤ 线路上限</td><td>与仓库整体处理能力是不同层面的限制。</td></tr>
<tr><td>碳排放预算</td><td>各线路“单位排放 × 运输量”之和 ≤ 总预算</td><td>把整张网络联系起来，形成成本与减排的权衡。</td></tr></table>
<p>报告第 3–4 页逐项给出符号、公式、业务含义和边界；第 11 页检查各类约束的满足程度和活跃数量。</p>
<h2>3. 为什么选择这些算法</h2>
<table><tr><th>方法</th><th>项目中的作用</th><th>必须讲清楚的边界</th></tr>
<tr><td>表式单纯形法</td><td>直观展示换基与最优顶点</td><td>大表和人工变量会增加开销。</td></tr>
<tr><td>修正单纯形法</td><td>比较维护整张表与分解基矩阵</td><td>这里每步重新做稠密 LU，不等同于成熟的稀疏求解器。</td></tr>
<tr><td>稀疏预测—校正内点法</td><td>实现牛顿方向、互补性与稀疏线性系统</td><td>这类网络约束行数较少，天然有利于它。</td></tr>
<tr><td>预条件重启 PDHG</td><td>研究现代一阶算法：稀疏乘法、对角步长、重启</td><td>重启是明确说明的启发式规则，不套用不适用的线性收敛定理。</td></tr>
<tr><td>OR-Tools PDLP</td><td>现代成熟求解器基准</td><td>单独标明是调用现有库，不冒充自行实现。</td></tr>
<tr><td>HiGHS</td><td>成熟 LP 基准与数值参照</td><td>再用其 IPM 模式交叉核对参照目标值。</td></tr></table>
<h2>4. 实际结果</h2>
<p>25 个最终测试实例中，稀疏内点法、PDLP、HiGHS 均达到统一的 10<sup>−6</sup> 误差标准；自行实现的预条件重启 PDHG 为 20/25。两种教学单纯形实现在全部 15 个小规模实例上成功；大规模按预先规定不运行稠密单纯形。</p>
<p>最大规模单个实例的三次运行中位时间如下。“成功”同时要求满足原模型约束和原始—对偶误差标准，不能只看成本数值。</p>
<table><tr><th>方法</th><th>运行时间</th><th>成功数</th><th>最终统一误差</th></tr>__ROWS__</table>
<img src="network/figures/network_scaling.png" alt="规模与运行时间">
<p>这组结果表明：现代算法值得研究，但是否更快要由问题结构决定。最大网络虽然有 32,400 个变量，内点法需要分解的系统却只有 511 行，所以稀疏内点法有明显的结构优势。PDLP 达到了精度要求，速度优势没有在本组算例中出现。未收敛的 PDHG 结果完整保留。</p>
<h2>5. 加分研究：消融与敏感性分析</h2>
<p>算法消融分别比较统一步长、对角预条件、预条件加重启，以及额外按右端值归一化的版本。在固定 10,000 步预算下，预条件加重启使五个诊断算例中的两个达到目标精度，其余三种版本均未达到。中位残差显示了具体改进，不能把它夸大成每个实例都会更快。</p>
<img src="network/figures/network_ablation.png" alt="消融比较">
<p>碳预算用 β 调整：β = 1 对应无碳约束的最低成本方案所需排放；β 越小越严格。五个模拟算例中，β = 0.1 的成本中位增幅为 <strong>__PREMIUM__%</strong>，β = 0.5 时为 <strong>__MAINPREMIUM__%</strong>。这说明减排预算会改变运输方案，也可通过影子价格估计进一步收紧预算的局部边际成本。</p>
<img src="network/figures/carbon_frontier.png" alt="碳预算与成本权衡">
<h2>6. 课堂讲解顺序（30 分钟）</h2>
<table><tr><th>时间</th><th>内容</th></tr><tr><td>0–4 分钟</td><td>业务场景、研究问题、为什么仍然是 LP。</td></tr><tr><td>4–10 分钟</td><td>变量、目标函数和六类约束，重点解释仓库平衡与碳预算。</td></tr><tr><td>10–18 分钟</td><td>四种自实现算法的基本更新、开销和停止条件。</td></tr><tr><td>18–24 分钟</td><td>小规模核对、大规模性能、失败案例与原因。</td></tr><tr><td>24–28 分钟</td><td>算法消融、成本—排放曲线、影子价格。</td></tr><tr><td>28–30 分钟</td><td>结论、局限与可复现材料。</td></tr></table>
<h3>常见提问</h3>
<p><strong>为什么不用神经网络或遗传算法？</strong>这个问题本身是 LP，具有明确的凸性与对偶证据；本项目的现代方法选择是 PDHG/PDLP，能直接利用稀疏结构并检验可行性和最优性。</p>
<p><strong>是不是只调用求解器？</strong>四种教学算法有自己的更新与终止逻辑，生产级 PDLP 和 HiGHS 单独作为外部基准。</p>
<p><strong>怎样知道解正确？</strong>保存所有最终原始与对偶变量，独立重新计算全部约束残差和有效对偶下界；390 组主实验输出全部完成核验，其中 375 组达到目标精度，15 组忠实保留为未达到精度。</p>
<p><strong>有真实企业数据吗？</strong>没有。这是可复现的模拟研究，成本与排放均为归一化参数，不作为真实企业经营结论。</p>
<h2>7. 代码与复现材料</h2>
<p>源码、模拟实例、原始结果、图表生成脚本和复现说明均在本仓库中。历史计时环境保存在 network/results/environment.json；本地目录已脱敏，数值结果保持原样。团队复现情况不改变归档计时的历史来源。</p>
<p>参考代码与文档准备使用了 AI 辅助。团队已报告复现代码；个人贡献应按实际工作记录，本仓库不分配个人独立实现署名。</p>
</main></html>'''
doc=doc.replace('__ROWS__',rows).replace('__PREMIUM__',f'{s["carbon_premium_beta01"]:.1f}').replace('__MAINPREMIUM__',f'{s["carbon_premium_beta05"]:.1f}')
(ROOT/'中文阅读说明.html').write_text(doc,encoding='utf-8')
print('Chinese guide written')
