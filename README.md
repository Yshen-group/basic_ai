# 人工智能基础课程作业

> [!IMPORTANT]
>
> a. 请以清晰、简洁的语言回答作业问题，力求逻辑严谨、表述准确。
>
> b. 如对作业有任何疑问，欢迎在平台上提出 issue 或直接联系[助教邮箱](mailto:xuchenlong796@gmail.com)，我们会及时提供帮助。
>
> c. 作业本身的要求之外，鼓励尝试更多的创新与探索，提出独特见解或设计新的应用场景，将有助于您深入理解相关内容。
>
> d. 请务必按照提交截止日期完成作业，逾期提交需提前与助教联系。

## 为什么存在这个仓库？

该仓库旨在为交通学院的人工智能导论课程提供作业问题和学习支持，帮助学生更好地掌握课程内容。通过集中整理和分享学习资料，师兄师姐们为师弟师妹们提供了一个便捷的学习平台，帮助新同学更快速地理解课程知识点、适应课程节奏。

仓库中不仅包含每次作业的详细说明，还推荐了一系列相关课程与参考资料，期望大家能够通过这些资源深入学习，并为将来的学习和研究打下坚实的基础。

|                           课程链接                           |                           课程简介                           |                           课程备注                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [CS 188：introduction to AI](https://inst.eecs.berkeley.edu/~cs188/fa24/) | **人工智能导论基础**，从行为流派介绍如何实现帮助人们工作的人工智能。从搜索问题、约束规划问题、马尔科夫决策过程、强化学习、机器学习、神经网络等内容 |  课程内容为科普性，内容简单，其中吃豆人小游戏作为本次作业1   |
| [CS 61A: Structure and Interpretation of Computer Programs](https://cs61a.org/) | **程序构造与运行原理**，课程地位类似于同济本科学习的程序设计，这里主要以 Python 出发介绍程序设计中的函数式编程、控制语句、类和对象等等 |            推荐观看并做完课程，提升自身的代码能力            |
| [CS 285：Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine老师的**深度强化学习**基础，从深度强化学习领域的基础知识到最先进展的延伸，非常值得学习 |                 难度较高，推荐观看和写完作业                 |
|    [CS229: Machine Learning](https://cs229.stanford.edu/)    | Andrew Wu（吴恩达）老师的**机器学习课程**，主要从概率统计和数学推导介绍机器学习中模型基本原理，包括但不限于广义线性模型、聚类模型、树模型、集成学习理论等 | 非常推荐观看，同时把提供的[课程作业](https://github.com/learning511/cs229-assignments)给做了 |
|           [《动手学深度学习》](https://zh.d2l.ai/)           | 李沐老师的**深度学习课程**，基于 PyTorch 编程为大家介绍卷积神经网络、循环神经网络、注意力机制带来的计算机视觉、自然语言处理中的常见模型原理 |           非常推荐观看，也记得关注李沐老师的 B 站            |
|                           优化课程                           |                             待定                             |                             待定                             |





## 问题1： [20 分] 搜索问题（Search problem）

本部分目录为[search]，[更详细的材料可以阅读](https://inst.eecs.berkeley.edu/~cs188/fa24/projects/proj1/)

**搜索问题**（Search problem）是已知一个问题的初始状态和目标状态，希望找到一个操作序列来是的原问题的状态能从初始状态转移到目标状态，搜索问题的基本要素：

a.   初始状态

b.   转移函数，从当前状态输出下一个状态

c.   目标验证，是否达到最终的目标状态

d.   代价函数，状态转移需要的代价

**搜索算法**（Search algorithm）指的是利用计算机的高性能来有目的穷举一个问题解空间的部分或者所有可能情况，从而求出对于问题的解的一种方法，常见的搜索算法包括枚举算法、深度优先、广度有限、A star算法、回溯算法、蒙特卡洛树搜索等等。搜索算法的基本过程：

a.   从初始或者目标状态出发

b.   扫描操作的算子，来得到候选状态

c.   检查是否为最终状态

​	a)    满足，则得到解

​	b)   不满足，则将新状态作为当前状态，进行搜索

### 1.1 问题描述与定义（Problem Definition）

请从身边选择一个实例，将其抽象为搜索问题，可以对示例吃豆人游戏的搜索特性进行参考，包括状态、动作和奖励。考虑搜索问题的基本要素，例如初始状态、目标状态、状态空间、以及路径代价和启发函数。在该实例中：

​	•	**状态**表示搜索过程中的不同位置或局面；

​	•	**动作**定义了在不同状态间移动或转换的可能操作；

​	•	**奖励**是对每一步行动的反馈，衡量选择该路径的优劣。

请运行以下命令以启动 Pacman 游戏并测试您的搜索策略：

```python
python pacman.py
```

例子：[*使用 BFS求解八皇后问题* ](https://leetcode.cn/problems/eight-queens-lcci/description/)

### 1.2 搜索算法实现（Implementation of Search Algorithms）

以下是每种算法的中文伪代码，基于 class SearchProblem 提供的 API，可供在 Search.py 中实现：



**深度优先搜索 （Deep Search First，DFS）**

```
定义 DFS(问题)：
    初始化栈 Stack，存入初始状态
    初始化空集合 visited，存储已访问状态
    
    当 Stack 不为空时：
        弹出当前状态 current_state
        
        如果 current_state 是目标状态：
            返回成功路径
        
        如果 current_state 不在 visited 中：
            将 current_state 加入 visited
            
            获取当前状态的所有可能的后续状态：
                对于每个后续状态 successor：
                    如果 successor 不在 visited 中：
                        将 successor 压入 Stack
                    
    返回失败路径（若没有找到目标状态）
```



**广度优先搜索 （Board Search First，BFS）**

```
定义 BFS(问题)：
    初始化队列 Queue，存入初始状态
    初始化空集合 visited，存储已访问状态
    
    当 Queue 不为空时：
        弹出当前状态 current_state
        
        如果 current_state 是目标状态：
            返回成功路径
        
        如果 current_state 不在 visited 中：
            将 current_state 加入 visited
            
            获取当前状态的所有可能的后续状态：
                对于每个后续状态 successor：
                    如果 successor 不在 visited 中：
                        将 successor 加入 Queue
                        
    返回失败路径（若没有找到目标状态）
```

**一致代价搜索 （Uniform Cost Search，UCS）**

```
定义 UCS(问题):
    初始化优先队列 PriorityQueue，存入初始状态（代价为0）
    初始化空集合 visited，存储已访问状态

    当 PriorityQueue 不为空时：
        弹出当前状态 current_state 及其代价 current_cost

        如果 current_state 是目标状态：
            返回成功路径

        如果 current_state 不在 visited 中：
            将 current_state 加入 visited

            获取当前状态的所有可能的后续状态：
                对于每个后续状态 successor 及其移动代价 step_cost：
                    如果 successor 不在 visited 中：
                        计算新代价：new_cost = current_cost + step_cost
                        将 successor 及 new_cost 加入 PriorityQueue

    返回失败路径（若没有找到目标状态）
```



**A\* 搜索（A Star Search）**

```
定义 A_star(问题)：
    初始化优先队列 PriorityQueue，存入初始状态（代价为启发值 h(初始状态)）
    初始化空集合 visited，存储已访问状态
    定义启发函数 h(state)，用于估计 state 到目标的距离

    当 PriorityQueue 不为空时：
        弹出当前状态 current_state 及其总代价 current_cost

        如果 current_state 是目标状态：
            返回成功路径

        如果 current_state 不在 visited 中：
            将 current_state 加入 visited

            获取当前状态的所有可能的后续状态：
                对于每个后续状态 successor 及其移动代价 step_cost：
                    如果 successor 不在 visited 中：
                        计算新总代价：new_cost = current_cost + step_cost + h(successor)
                        将 successor 及 new_cost 加入 PriorityQueue

    返回失败路径（若没有找到目标状态）
```



可以运行下列代码可以观察算法效果

```python
python pacman.py -l mediumMaze -p SearchAgent -a fn=dfs
```

### 1.3 智能体构建（Agent Construction for Search）
以下是实现吃豆人游戏 Corner Problem 的一些指导，帮助您在完成搜索算法基础上修改 SearchAgents.py 来解决这个问题：

**1. 完善 Corner Problem**

Corner Problem 是一个典型的搜索问题，吃豆人需要经过四个角落并吃掉所有豆子。我们需要定义一个新的 CornerProblem 类来描述这个问题，继承 SearchProblem 并实现以下内容：

​	•**状态表示**：每个状态需要包含吃豆人当前的位置，以及哪些角落的豆子已被吃掉。

​	•**目标测试**：检查是否已访问所有角落，即是否已吃掉所有角落的豆。

​	•**状态转换**：定义从当前位置到下一个位置的所有可能动作。

​	•**成本计算**：每一步的成本为 1（可以根据实际情况调整）。

**2. 利用 DFS 吃掉角落的豆**

​	在完成 CornerProblem 的定义后，可以使用 DFS 实现吃掉四个角落的豆子。可以通过在 SearchAgents.py 中创建 CornerSearchAgent 类，该类调用 DFS 以找到访问所有角落的路径：

**3. 利用启发式（Heuristic）优化**

在 DFS 的基础上，可以使用 A* 搜索来提高算法效率。为此，需要定义启发函数 cornerHeuristic，将此启发函数传递给 A* 搜索以优化 Corner Problem 的求解：

​	•启发函数可以为计算吃豆人从当前位置到所有未访问角落的最小曼哈顿距离和。

### 1.4 加分项：吃掉所有豆子！（Bonus: Eat All the Dots!）



## 问题2： [20 分] 机器学习（Machine Learning）

本章节对应目录为 p2_machine_learning

### 2.1 异常检测（Anomaly detection）

异常检测是数据分析中的一个重要任务，旨在识别不符合数据整体规律的样本。机器学习通过从正常数据中学习其规律和分布特征，构建模型来自动识别异常点。机器学习的异常检测可以分为**监督学习**、**无监督学习**和**半监督学习**三种模式

1. **监督学习**：需要有标记的异常和正常样本数据，训练模型以识别未来数据中的异常点。这种方法在标记数据丰富时效果较佳，但标记异常数据通常较困难。
2. **无监督学习**：不需要标记数据，仅利用正常数据的分布特点识别异常点，这在大多数实际场景中更为常见。
3. **半监督学习**：只使用正常样本构建模型，以此为基准来判断未来样本是否为异常。

常见的异常检测模型包括，具体原理请自行查阅相关资料：

1. **高斯模型（Gaussian Mixture Model, GMM）**
2. **孤立森林（Isolation Forest）**
3. **基于密度的方法（Density-Based Methods）**
4. **主成分分析（Principal Component Analysis, PCA）**
5. **支持向量机的单类分类（One-Class SVM）**
6. **自编码器（Autoencoder）**

其中数据说明

| 变量      | 注释         |
| --------- | ------------ |
| X.npy     | 待标记点坐标 |
| X_val.npy | 已标记点坐标 |
| yval.npy  | 已标记点标签 |

<img src="https://chenxia31blog.oss-cn-hangzhou.aliyuncs.com/blog/image-20241023153305355.png" alt="数据展示" style="zoom: 33%;" />

### 2.2 离散选择模型 （DCM）

**离散选择模型**是一种用于建模决策者在有限多个选项中进行选择的过程的统计工具。该模型基于效用最大化理论，假设每个可选项都带有一定的效用值，且决策者会偏好带来最高效用的选择。离散选择模型通常用于模拟和分析个体的选择行为，其典型模型包括：

- **二元选择模型**：适用于两个选项的场景，常见的模型有 Logit 和 Probit 模型。二元选择模型通常用于分析二选一的决策过程，适用场景包括选择购买或不购买、选择乘坐公交或步行等。
- **多项选择模型**：用于多个选项的场景，其中最常见的是多项 Logit 模型。该模型通过估计不同选项的效用差异，解释个体在多个选项间如何选择。应用场景包括分析不同交通方式的选择（如开车、公交、自行车）、商品偏好等。

离散选择模型广泛应用于交通运输、市场营销、经济学等领域，通过量化影响因素来预测个体的选择行为，为政策制定、产品设计等提供科学依据。



本次作业中给定的 **Dataset** 中包含了上海定制公交线路开关的面板数据，字段信息可参考[该论文](https://www.sciencedirect.com/science/article/pii/S2046043023001077)，其中 label 字段表示线路的开关状态。请建立多个模型，对比分析各字段对线路开关状态的影响因素。

1. 利用离散选择库 [Biogeme](https://biogeme.epfl.ch/)构建合适的逻辑回归模型，分析特征显著性
2. 基于树模型构建特征重要性的解释，例如Ada boost、[LightGBM](https://lightgbm.readthedocs.io/en/stable/)、[XGBoost](https://xgboost.readthedocs.io/en/stable/)、[CatBoost](https://catboost.ai/) 等
3. 利用因果理论分析因果效应，至少实现以下一种方法：基于约束的因果关系发现或基于 Null importance 的因果反驳。

### 2.3 牛顿法迭代实现 （Newton）

**牛顿法**（Newton’s method），又称**牛顿-拉弗森方法**（Newton-Raphson method），是一种用于求解实数域或复数域方程近似解的迭代方法。牛顿法的核心思想是利用函数在接近零点附近的切线来逼近零点，并通过反复迭代逐渐收敛到方程的根。具体步骤如下：

- **初始点选择**：选择一个接近函数零点的初始值 
- **切线迭代更新**：在每次迭代中，使用当前点的切线斜率（即导数）来更新 x 值。公式为 <img src="https://chenxia31blog.oss-cn-hangzhou.aliyuncs.com/blog/image-20241023164352151.png" alt="牛顿发公式" style="zoom:50%;" />
- **收敛**：不断迭代上述过程，直到 x 收敛至零点，满足给定的精度要求。

完成 new_ton 脚本中的 update 函数逻辑，同时增加可视化过程



## 问题3： [30分] 深度学习（Deep learning）

本章节对应目录为 `deeplearning`，更多详细信息可以参考阅读：https://zh-v1.d2l.ai/index.html

深度学习是人工智能和机器学习的一个分支。其主要希望利用多层的人工神经网络来实现特定的任务目标，包括但不限于目标检测、语音识别、语音翻译、文本生成、图文匹配等功能。其优点是可以自动的是图像、视频、文本、音频中自动的学习特征，而不需要引入人类领域的知识。

在深度学习项目中，完整的构建流程确保了模型的高效训练与评估。以下是主要流程概述：

- **数据准备**  收集、清洗数据（Data Collection and Cleaning），并执行标准化、归一化等预处理，确保数据适合模型学习。

- **数据加载器**  使用 `DataLoader` 实现批量加载、数据打乱（Batch Loading and Shuffling），以优化训练效率。

- **模型定义**  设计网络架构（Network Architecture Design）并定义前向传播（Forward Pass），根据任务需求选择合适的层次、激活函数等。

- **损失函数与优化器**  选择适合的损失函数（如交叉熵 Cross-Entropy 或均方误差 Mean Squared Error）和优化器（如 SGD、Adam），用于模型参数更新（Parameter Optimization）。

- **训练与指标监控**  执行训练循环（Training Loop），包括前向传播、反向传播（Backward Pass）和优化步骤，监控损失值、准确率（Accuracy）等指标，判断模型收敛情况。

- **验证与测试**  通过验证集（Validation Set）定期评估模型，防止过拟合，并在测试集（Test Set）上最终评估模型的泛化能力（Generalization Performance）。

- **模型保存与加载**  保存模型参数（Model Saving），确保在训练中断后可恢复训练或用于推理（Inference）。

- **模型部署**  优化推理性能（Inference Optimization），将模型集成至实际系统，支持自动化预测与决策。

  

### 3.1 计算机视觉（Computer Vision）

在 `CV` 文件夹中展示了如何使用 MLP 和 CNN 实现手写数据集的识别任务。本次作业要求您进一步优化并拓展此任务的内容，具体要求如下：

1. **数据降维与特征解释**  

  从数据层面的角度分析主成分分析（PCA）对 MLP 和 CNN 的适用性，并解释原因：

  \- 对于 MLP：PCA 是否有助于提升模型的特征提取效果？如果有，其背后的原因是什么？

  \- 对于 CNN：PCA 在卷积网络中是否同样有效？请解释卷积操作是否影响 PCA 在 CNN 中的应用。

2. **自定义卷积模型**

  基于 AlexNet 的网络架构[《ImageNet Classification with Deep Convolutional Neural Networks》](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)，设计一个自定义卷积神经网络模型并完成以下任务：

  \- 模仿 AlexNet 设计网络层的结构和配置，并说明每层的功能及对图像特征的提取方式。

  \- 解释 AlexNet 如何解决“深度卷积”网络的学习难题，例如通过什么技术缓解了梯度消失、训练速度慢等问题，并分析该设计对您的模型构建的影响。

3. **效果对比分析**

  \- 对比分析 MLP 与 CNN 在手写数据集识别任务上的性能，包括训练精度、收敛速度和泛化能力等。

  \- 结合不同模型的架构特点，探讨在手写数据集上各自的优缺点，讨论在不同条件下模型的适用性。

> **提示**   您可以使用 [PyTorch](https://pytorch.org/docs/stable/index.html) 替代原先的 [TensorFlow](https://www.tensorflow.org/api_docs) 代码进行模型实现与训练。请在报告中清晰展示代码实现，并以图表或表格形式展示不同模型的性能对比。



### 3.2 自然语言处理（Natural language processing）
在 `NLP` 文件夹中的 `prebert.ipynb` 中，基于 `transformer` 库的 BERT 微调方案已初步完成，用于情绪识别任务（参考论文 [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)）。数据集处理和模型训练部分已完善，本次作业需要您在此基础上进一步完成以下任务：

1. **模型修改与二分类输出层设计**

  \- 在现有的 BERT 模型基础上增加一层 MLP，以满足情绪识别的二分类任务需求。您需要修改 `BertSST2Model`，以便模型输出符合二分类要求。

  \- 为实现此任务，请参考 [PyTorch 模块](https://pytorch.org/docs/stable/index.html) 的使用文档，以理解如何自定义网络层并将其集成到 BERT 模型中。

2. **训练过程与模型保存**

  \- 编写完整的模型训练过程，包括前向传播、反向传播和优化步骤，并实现训练过程中的定期模型保存功能。

  \- 在训练循环中，设置检查点机制以保存模型参数，确保在训练中断或完成后可以恢复模型或进行推理。

3. **BERT 损失函数的解释**

  \- 介绍 BERT 模型在二分类任务中通常使用的损失函数，并说明其工作原理。

  \- 阐述为什么不选择均方误差（MSE）作为损失函数，重点分析分类任务中交叉熵损失（Cross-Entropy Loss）的优势和适用性。

> **提示**   在报告中详细记录您的实现过程和代码，并以注释说明关键部分的功能。对于模型训练和保存功能，请确保代码清晰、易于复用。



### 3.3 生成对抗网络（Generative adversarial network）

本作业要求利用生成对抗网络（GAN）生成动漫头像，参考论文 [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)。数据集可从[此链接](https://LTAI5tNwHtQGucyrs15ssbvo@chenxia31blog.oss-cn-hangzhou.aliyuncs.com/fileshare/datasets.zip)下载。请在实验中完成以下任务：

1. **GAN 文件目录理解与训练结果绘制**

  \- 理解 `GAN` 目录中的文件结构和代码模块。

  \- 训练模型并保存生成的动漫头像图片，绘制并展示训练结果随迭代变化的效果图，以帮助观察模型的生成质量和稳定性。

2. **解释维度坍塌（Mode Collapse）问题与模型改进**

  \- 解释 GAN 模型中可能出现的**维度坍塌（Mode Collapse）**现象，分析其对生成质量的影响。

  \- 在此基础上，尝试对 baseline 模型进行调整以减轻维度坍塌的问题，探索可能的改进方案（例如，使用不同的损失函数、加入正则化技术或引入生成器和判别器的结构调整等）。

> **提示**   请在报告中展示您对 GAN 原理和文件结构的理解过程，并详细记录模型训练和改进的过程。确保生成结果图清晰，以便于分析和评价模型的表现。



## 问题4：[20 分] 深度强化学习（Deep reinforcement learning）

深度强化学习（Deep Reinforcement Learning, DRL）结合了强化学习和深度学习的优势，旨在训练智能体（agent）在与环境（environment）交互中不断学习策略，以最大化预期收益。DRL 在复杂的高维环境中应用深度神经网络（DNN）作为函数逼近器，使智能体能够从原始观测数据中直接学习出有效的策略。这一方法在机器人控制、游戏AI、自动驾驶等领域展现出强大的决策和适应能力。更多可以参考 CS185 课程

### 4.1 论文精读（Paper Review）

DQN（深度 Q 网络）是经典的深度强化学习算法，针对 DQN 的诸多改进极大提升了其性能。为加深理解，建议阅读论文 [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)，此文整合了 DQN 的多项增强技术，形成了更高效的强化学习算法。

要求阅读和理解 Rainbow 的改进措施，以获得对深度强化学习更全面的认识。

### 4.2 基准实现（Baseline Implementation）
在处理视频任务时，状态通常以图像帧的形式提供，因此可以通过将状态图像传入 CNN 模块进行特征提取，从而减少模型的参数量并提升网络的泛化性能。为此，我们需要对 Q-Network 进行以下修改：

**引入卷积层（CNN）**

在 Q-Network 的前端添加卷积层，以便处理输入的图像数据。卷积层能够有效提取图像中的空间特征，同时避免传统全连接层的高参数量问题。典型的 CNN 模块包括卷积层、池化层和非线性激活函数（如 ReLU），这些层将逐步提取图像的高维特征。

**特征映射到全连接层**

在卷积层之后，将卷积得到的特征图（Feature Map）展平，并传递到全连接层。全连接层将这些特征用于学习 Q 值函数，帮助智能体在特定状态下对不同动作进行价值估计。

**修改 Q 值预测层**

最终的输出层需要根据动作空间的大小来调整，以输出每个可能动作的 Q 值。若是离散动作空间，则输出的维度应等于动作空间的数量。

<img src="https://chenxia31blog.oss-cn-hangzhou.aliyuncs.com/blog/image-20241023182004929.png" alt="Lunar landing" style="zoom:50%;" />



## 问题5：[10 分] 运筹优化（Operation  research）

运筹优化是一门应用数学的分支，主要用于构建和求解复杂的决策问题，广泛应用于科学研究、工程设计和商业决策等领域。运筹优化利用数学建模技术，将实际问题简化为优化模型，帮助决策者在约束条件下优化资源配置与系统性能。

[Pyomo (Python Optimization Modeling Objects)](https://pyomo.readthedocs.io/en/stable/) 是一种功能强大的 Python 优化建模工具，用于构建和求解复杂的优化模型。Pyomo 提供了丰富的建模对象库，支持线性、非线性、整数和混合整数等优化问题，使用户可以方便地进行建模、求解和结果分析。Pyomo 在学术和工业应用中得到广泛使用。



在运筹优化中，求解方法主要分为两类：**精确求解**和**近似求解**。

**精确求解**方法旨在找到问题的全局最优解，适用于问题规模较小或结构良好的情况。这类方法能够保证解的最优性，但随着问题规模增大，求解时间可能显著增加。常见的精确求解方法包括：

- **线性规划**（Linear Programming, LP）：用于求解线性目标函数和线性约束条件的优化问题。

- **整数规划**（Integer Programming, IP）：适合变量必须是整数的优化问题，通常使用分支定界法（Branch and Bound）进行求解。

- **动态规划**（Dynamic Programming, DP）：用于分阶段决策问题，适合求解具有递归性质的最优子结构问题。

常用的精确求解器包括 CPLEX、Gurobi 和 CBC，它们能够处理大规模线性和非线性优化模型，为小型和中型问题提供最优解。

- [GLPK](https://www.gnu.org/software/glpk/)（GNU Linear Programmed Kit）是GNU维护一个线形规划工具包，对于求解大规模的额线性规划问题速度缓慢
- [CPLEX](https://www.ibm.com/cn-zh/products/ilog-cplex-optimization-studio)，是IBM开发的商业线性规划求解器，可以求解LP,QP,QCQP,SOCP等四类基本问题和对应的MIP，社区版仅支持低于1000变量的使用，教育版无限制，建模时间长，但是求解速度非常快
- [Gurobi](https://www.gurobi.com/)，是CPLEX团队创始人重新创建的商业求解器，和cplex速度差别不大，相对来说比Cplex好一点
- [OR-Tools](https://github.com/google/or-tools)，谷歌开发的商业求解



**近似求解**方法通常应用于 NP-hard 问题或超大规模优化问题，能够在合理时间内找到接近最优的解。尽管不保证全局最优性，但这类方法通常能提供可接受的解，适用于复杂的大规模问题。常见的近似求解方法包括：

- **遗传算法**（Genetic Algorithm, GA）：基于自然选择和遗传机制的随机搜索算法，适用于多峰问题。

- **模拟退火**（Simulated Annealing, SA）：利用物理退火过程的启发，随机搜索解空间，避免局部最优。

- **粒子群优化**（Particle Swarm Optimization, PSO）：基于群体协作行为，用于解决连续和离散优化问题。

这些方法广泛应用于物流优化、生产调度和路径规划等领域，能够在可接受的时间内获得接近最优的解。

- [GeatPY](https://github.com/geatpy-dev/geatpy)  高性能的Python遗传和进化算法工具箱。

#### 5.1 投资优化（精确求解）

假设你有 100 万美元要投资于三种资产（资产 1、资产 2 和资产 3），每种资产的预期回报率和风险如下表所示，

|        | 预期回报率 | 风险系数 |
| ------ | ---------- | -------- |
| 资产 1 | 10%        | 0.5      |
| 资产 2 | 12%        | 0.7      |
| 资产 3 | 8%         | 0.4      |

目标是在总投资回报率不低于 105 的情况下，最小化总投资的风险，要求使用 Pyomo 建立并调用求解器优化该问题，输出每种资产的投资金额和最小化的风险值。

#### 5.2 TSP 问题（近似求解）

在 TSP（旅行商问题）中，我们希望找到一条访问所有城市的最短路径。利用 GeatPy 的启发式进化算法可以有效地求解 TSP 问题，其逻辑过程如下：

1. **初始化初始种群**   随机生成多个初始解，每个解代表一个城市访问顺序，从而构成初始种群。

2. **迭代优化**   通过多次迭代不断优化种群，直到满足终止条件（例如，达到最大迭代次数或解的收敛）。每次迭代包含以下步骤：

  \- **适应度计算**：计算种群中每个个体的适应度，其中适应度为路径的总长度（目标是最小化该值）。

  \- **进化操作**：在种群中执行选择、交叉和变异操作，以生成下一代具有更优解的种群。

  \- **更新种群**：将生成的下一代种群替换当前种群，并继续下一轮迭代。

3. **终止条件**  当达到指定的迭代次数或种群适应度收敛到最小路径时，算法终止，输出最优路径。

该方法通过进化算法逐步优化路径长度，以获得接近最优的解，目标是输出总路径最小化的城市访问顺序。

<img src="https://chenxia31blog.oss-cn-hangzhou.aliyuncs.com/blog/image-20241023181655512.png" alt="TSP 问题求解" style="zoom:33%;" />
