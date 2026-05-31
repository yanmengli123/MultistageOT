from abc import ABC, abstractmethod
"""
MultistageOT: 多阶段最优传输轨迹推断框架

核心功能：
- 从单细胞RNA测序数据推断细胞分化轨迹
- 计算伪时间（pseudotime）排序
- 预测细胞命运概率

作者: Dahlin Lab
论文: PNAS 2025 - Multistage optimal transport infers trajectories from a snapshot of single-cell data
"""

import pandas as pd
import numpy as np
import time
import copy
import os
import sys
import scipy
from datetime import timedelta

# CHANGELOG: Updated 20250124 with updates that updated doc-comments and removed unused
#           methods, attributes and variables.


class CellGraph:
    """ 细胞图类 - 基于转换概率矩阵构建马尔可夫链模型

    细胞图将MultistageOT计算得到的运输计划转换为细胞间的转换概率矩阵，
    用于后续的马尔可夫链分析（如计算吸收概率、命运预测等）。

    Parameters:
    -----------
    transition_matrix : numpy array (2D matrix)
        N x N 的转换概率矩阵，描述细胞状态间的转换概率

    Attributes:
    -----------
    adjacency_matrix : numpy array
        四舍五入的转换矩阵（用于邻接关系）
    transition_matrix : numpy array
        原始转换概率矩阵
    log_transition_matrix : numpy array
        对数转换概率矩阵（用于最大似然估计）
    """
    def __init__(self, transition_matrix):
        # 保留一位小数的邻接矩阵（用于图结构分析）
        self.adjacency_matrix = np.round(transition_matrix, 1)
        # 原始转换概率矩阵
        self.transition_matrix = transition_matrix
        # 对数变换的转换矩阵（加小常数避免log(0)）
        self.log_transition_matrix = np.log(transition_matrix + 1e-32)





class OTModel(ABC):
    """ 最优传输模型基类（抽象类）

    定义所有OT模型共有的接口：
    - fit(): 拟合模型
    - save_model(): 保存模型
    - load_model(): 加载模型

    Attributes:
    -----------
    couplings : numpy array
        基于最优传输计划的数据点之间耦合强度
    transport_plans : list
        最优传输计划列表（每个阶段一个矩阵）
    transport_costs : list
        每个传输计划的运输成本
    data : pandas DataFrame
        拟合模型所用的数据
    """
    def __init__(self):
        # 耦合强度（基于最优传输计划的数据点之间）
        self.couplings = None
        # 最优传输计划列表
        self.transport_plans = []
        # 每个传输计划的成本
        self.transport_costs = []
        # 拟合模型所用的数据
        self.data = None

    # 抽象方法：子类必须实现
    @abstractmethod
    def fit(self):
        """ 拟合模型到数据 """
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        """ 保存模型到磁盘 """
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        """ 从磁盘加载模型 """
        raise NotImplementedError



class MultistageOT(OTModel):
    """ 多阶段最优质量传输模型 (MultistageOT)

    从单细胞数据的快照中推断细胞的时间排序和关联性。

    模型图示：
                             o  (子细胞/daughter cell)
                            /
            (父细胞/parent cell) (*) -- o  (子细胞/daughter cell)

    分化过程从初始状态（initial_cells）开始，通过多个中间阶段，
    最终到达终端状态（terminal_cells）。

    Parameters:
    ----------
    initial_cells : list
        初始边际（initial marginal）对应的细胞索引列表，即最不成熟的细胞（如干细胞）
    terminal_cells : list
        终端边际（terminal marginal）对应的细胞索引列表，即最成熟的细胞（如系定向细胞）
    n_groups : int
        中间边际组的数量，即建模从初始细胞分化到终端细胞所需的最大步数
    fate_groups : list, optional
        每个终端命运组的标签列表（默认=None）
    auxiliary_cell_cost : float, optional
        辅助细胞成本，用于处理离群值时的惩罚参数（默认=None）
    epsilon : float, optional
        熵正则化参数，控制推断的细胞-细胞耦合的扩散水平（默认=None）

    Example:
    --------
    # 定义初始和终端细胞
    initial_cells = [1, 2, 3]   # 根细胞（干细胞）的索引
    terminal_cells = [7, 8, 9]   # 终端细胞的索引
    T = 21                       # 最终"时间点"
    epsilon = 0.01               # 正则化参数

    # 创建模型实例
    msot = MultistageOT(
        initial_cells=initial_cells,
        terminal_cells=terminal_cells,
        n_groups=T-1,            # 中间阶段数
        epsilon=0.015
    )

    # 运行.fit()方法找到最优耦合
    msot.fit(data)

    # 之后可以使用：
    # - msot.pseudotemporal_order() 获取伪时间排序
    # - msot.cell_fate_probabilities() 获取命运概率
    """
    

    def __init__(self,
                 initial_cells: list = None,
                 terminal_cells: list = None,
                 n_groups: int = None,
                 fate_groups: list = None,
                 auxiliary_cell_cost: float = None,
                 epsilon: float = None):
        """初始化MultistageOT模型

        Args:
            initial_cells: 初始细胞索引列表（最不成熟的细胞，如干细胞）
            terminal_cells: 终端细胞索引列表（最成熟的细胞，如终末分化细胞）
            n_groups: 中间运输阶段的数量（分化的最大步数）
            fate_groups: 命运组标签列表
            auxiliary_cell_cost: 辅助细胞成本（用于处理离群值）
            epsilon: 熵正则化参数
        """

        # ==================== 私有属性 ====================
        # 阶段数量
        self._NUM_GROUPS = n_groups
        # 初始正则化参数
        self._EPSILON = epsilon

        # ==================== 近端Sinkhorn方案私有属性 ====================
        self._PROXIMAL_EPSILON = None  # 近端正则化参数
        self._PROXIMAL_EPSILON_HISTORY = []  # 近端epsilon历史记录
        self._TOTAL_EPSILON_HISTORY = []  # 总epsilon历史记录
        self._INNER_ITERATIONS = []  # 内层迭代次数
        self._OUTER_PROXIMAL_ITERATIONS = 0  # 外层近端迭代次数
        self.__PRIOR = None  # 先验运输计划（用于近端方案）
        self.__USE_PRIOR = None  # 是否使用先验

        # ==================== 辅助细胞相关 ====================
        # 辅助细胞成本，用于在模型中引入"离群点吸收"机制
        # 当设置此值时，模型会创建额外的辅助状态来吸收异常细胞
        self._AUXILIARY_CELL_COST = auxiliary_cell_cost

        # 转换矩阵（马尔可夫链）
        self._TRANSITION_MATRIX = None

        # ==================== 公有属性 ====================
        self.initial_cells = initial_cells  # 初始细胞索引
        self.intermediate_cells = None  # 中间细胞索引（fit时自动计算）
        self.terminal_cells = terminal_cells  # 终端细胞索引
        self.fate_groups = fate_groups  # 命运组
        self.median_cost = None  # 代价矩阵的中位数（用于归一化）

        # ==================== 优化相关变量 ====================
        self.dual_variables = []  # 对偶变量（拉格朗日乘子）
        self.utility_variables = [[], [], 0]  # utility变量 [u列表, v列表, s]
        self.history = None  # 收敛历史记录




    def fit(self, data, verbose=True,
            log=False,
            patience=1,
            tolerance=1e-8,
            prior=None,
            sparse=False,
            checkpoints=None,
            path_to_checkpoints=None):
        """运行MMOT算法找到数据中的最优耦合

        执行多阶段最优传输的Sinkhorn迭代算法，通过块坐标上升法求解对偶问题。
        算法会迭代直到满足收敛条件（最大步长 + 不可行性 <= tolerance）

        更新以下公有属性:
            transport_plans   - 最优传输计划列表
            transport_costs  - 每个传输计划的成本
            dual_variables    - 对偶变量（拉格朗日乘子）
            history           - 收敛历史（max_steps和infeasibility）
            checkpoints       - Sinkhorn方案中的检查点数量

        Args:
            data : pandas DataFrame
                细胞数据，行=细胞，列=特征（如基因表达）
            verbose : bool
                是否打印进度信息（默认True）
            log : bool
                是否记录日志（默认False）
            patience : int
                两次可行性检查之间的迭代次数（默认1）
            tolerance : float
                收敛容忍度：当 max_step + infeasibility <= tolerance 时停止迭代（默认1e-8）
            prior : list, optional
                先验运输计划，用于 proximal Sinkhorn 方案（默认None）
            sparse : bool
                是否使用稀疏矩阵实现（默认False）
            checkpoints : int, optional
                保存检查点的频率
            path_to_checkpoints : str, optional
                保存检查点的目录路径

        Returns:
            None
        """
        # 设置verbose标志
        self._VERBOSE = verbose

        # 设置其他私有属性
        self._PATIENCE = patience
        self._TOLERANCE = tolerance
        self._SPARSE = sparse

        # 保存数据
        self.data = data

        # 自动识别中间细胞（不属于initial_cells或terminal_cells的细胞）
        self._set_intermediate_cells()

        # 保存检查点相关设置
        self._path_to_checkpoints = path_to_checkpoints
        self._checkpoints = checkpoints

        # 计算各类细胞的数量（考虑是否使用辅助细胞）
        # 如果使用辅助细胞，每个类别多一个"槽位"用于吸收离群点
        self._n0 = len(self.initial_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.initial_cells)
        self._n = len(self.intermediate_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.intermediate_cells)
        self._nF = len(self.terminal_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.terminal_cells)

        # ==================== 定义常向量 ====================
        self._1n0 = np.ones((self._n0, 1))  # 初始细胞个数的全1向量
        self._1n0_trans = self._1n0.T
        self._1n = np.ones((self._n, 1))    # 中间细胞个数的全1向量
        self._1n_trans = self._1n.T
        self._1nT = np.ones((self._nF, 1)) # 终端细胞个数的全1向量
        self._1nT_trans = self._1nT.T
        self._0n0 = np.zeros((self._n0, 1))
        self._0n = np.zeros((self._n, 1))
        self._0nT = np.zeros((self._nF, 1))

        # 定义最终"时间"：T = n_groups + 1
        # 例如：n_groups=10，则有 0, 1, 2, ..., 10, 11 共12个时间点
        self._T = self._NUM_GROUPS + 1

        # ==================== 运行Sinkhorn迭代 ====================
        if prior is not None:
            # 有先验：使用近端Sinkhorn方案
            self.__PRIOR = prior
            self.__USE_PRIOR = True
            self._sinkhorn_iterations_with_prior()
        else:
            # 无先验：使用标准Sinkhorn方案
            self._sinkhorn_iterations()

        return



    def _converged(self, infeasibility, max_step):
        """判断Sinkhorn算法是否收敛

        收敛条件：max_step + infeasibility <= tolerance
        其中：
        - max_step：所有对偶变量更新的最大值
        - infeasibility：边际约束违背的最大值

        Args:
            infeasibility: 不可行性度量（边际约束违背）
            max_step: 最大对偶变量更新步长

        Returns:
            bool: True表示收敛，False表示继续迭代
        """
        if (max_step + infeasibility) <= self._TOLERANCE:
            return True
        else:
            return False

    def _set_intermediate_cells(self):
        """自动识别中间细胞

        中间细胞 = 不在initial_cells也不在terminal_cells中的所有细胞
        这些细胞处于分化过程的中间阶段
        """
        self.intermediate_cells = self.data.loc[~self.data.index.isin(
            self.initial_cells + self.terminal_cells
        )].index.tolist()
        return


    def _get_C(self):
        """ Utility method for computing the squared Euclidean distance between
        marginals (i.e. the pairwise cost between 'cells')"""


        start_data = self.data.loc[self.initial_cells].values
        mid_data   = self.data.loc[self.intermediate_cells].values
        end_data   = self.data.loc[self.terminal_cells].values

        mid_end_data = np.concatenate((mid_data, end_data), axis = 0)
        C_0 = scipy.spatial.distance.cdist(start_data, mid_data, 'sqeuclidean')   # Costs between initial and intermediate
        C  = scipy.spatial.distance.cdist(mid_data, mid_end_data, 'sqeuclidean') # Costs between intermediate and intermediate
        C_T_1  = scipy.spatial.distance.cdist(mid_data, end_data, 'sqeuclidean') # Costs between intermediate and terminal
        

    
        median = np.median(C_0.flatten().tolist()+C.flatten().tolist()+C_T_1.flatten().tolist())
        self.median_cost = median
        
        C_0 = C_0/median  
        C   = C/median 
        C_T_1 = C_T_1/median 

        #Modify C if mass pool is used:
        if (self._AUXILIARY_CELL_COST is not None):
            C_0     = np.block([[C_0, np.inf*np.ones((self._n0-1,1)), self._AUXILIARY_CELL_COST*np.ones((self._n0-1,1))], 
                                [self._AUXILIARY_CELL_COST*np.ones((1,self._n+1))]])
            
            C_tilde = C[:,:self._n-1]
            C_hat   = C_T_1
            
            C_tilde = np.block([[C_tilde, self._AUXILIARY_CELL_COST*np.ones((self._n-1,1))],
                            [self._AUXILIARY_CELL_COST*np.ones((1,self._n-1)), np.zeros((1,1))]])

            C_hat   = np.block([[C_hat, self._AUXILIARY_CELL_COST*np.ones((self._n-1,1))],[np.inf*np.ones((1,self._nF-1)), self._AUXILIARY_CELL_COST*np.ones((1,1))]])

            C = np.block([C_tilde, C_hat])
            
            C_T_1 = C_hat
            

        #Prohibit self-couplings by giving infinite cost:
        C[:,:self._n] += np.diag(np.ones(self._n)*np.inf)


        return [C_0, C, C_T_1]

    def _retrieve_plans(self, K_0, K, K_hat, u, v, s):
        """从对偶变量计算传输计划矩阵

        根据Sinkhorn算法的最优传输公式 M = K * (u * v')，计算每个时间步骤的传输计划矩阵。
        该方法是标准Sinkhorn方案的核心函数，用于在每次迭代后恢复完整的传输计划。

        算法原理：
        - Gibbs核矩阵K通过 exp(-C/epsilon) 计算，其中C是代价矩阵
        - 对偶变量u和v通过Sinkhorn迭代更新
        - 传输计划M通过核矩阵与对偶变量的外积得到

        Args:
            K_0: 初始到中间的核矩阵 (n0 x n)，由 exp(-C_0/epsilon) 计算得到
            K: 中间到中间的核矩阵 (n x (n+nF))，由 exp(-C/epsilon) 计算得到
            K_hat: 中间到终端的核矩阵 (n x nF)，由 exp(-C_T_1/epsilon) 计算得到
            u: 对偶变量列表 [u_0, u_1, ..., u_T]，每个u_t是对应时间点的行缩放向量
            v: 对偶变量列表 [v_0, v_1, ..., v_T]，每个v_t是对应时间点的列缩放向量
            s: 缩放向量，用于控制中间细胞的质量守恒

        Returns:
            M_list: 传输计划列表 [M_0, M_1, ..., M_T]
                - M_0: 初始到中间的传输计划 (n0 x n)
                - M_1 到 M_{T-1}: 中间各阶段传输计划
                - M_T: 中间到终端的传输计划 (n x nF)
        """
        T = self._T

        M_0 = K_0 * (np.outer(u[0],v[1]))
        M_list = [M_0]

        for t in range(1, T-1):
            if (t == 1) & (self._AUXILIARY_CELL_COST is not None):
                M_list.append( K * ( np.outer(u[t][:-1]*s, np.vstack((v[t+1],v[T])) ) ) )
            else:
                M_list.append( K * ( np.outer(u[t]*s, np.vstack((v[t+1],v[T])) ) ) )

        M_list.append( K_hat * np.outer(u[T-1]*s, v[T]) )

        return M_list
    
    def _retrieve_plans_proximal(self, K_0, K, u, v, s):
        """从对偶变量计算近端方案的传输计划矩阵

        针对近端Sinkhorn方案的传输计划计算，与标准方案的区别在于：
        - 近端方案使用 G = K * prior 作为核矩阵，其中prior是前一次计算的传输计划
        - 核矩阵K变为列表形式，因为每个时间步骤可能使用不同的正则化参数

        该方法用于 proximal_sinkhorn 迭代过程中，根据更新后的对偶变量恢复传输计划。

        Args:
            K_0: 初始到中间的近端核矩阵 G_0 = K_0 * prior_0
            K: 近端核矩阵列表 [G_0, G_1, ..., G_{T-1}]，每个元素是对应阶段的自适应核矩阵
            u: 对偶变量列表 [u_0, u_1, ..., u_T]
            v: 对偶变量列表 [v_0, v_1, ..., v_T]
            s: 缩放向量

        Returns:
            M_list: 传输计划列表 [M_0, M_1, ..., M_T]
                - M_0: 初始到中间的传输计划
                - M_1 到 M_{T-1}: 中间各阶段传输计划
                - M_T: 中间到终端的传输计划
        """
        T = self._T

        M_0 = K_0 * (np.outer(u[0],v[1]))
        M_list = [M_0]

        for t in range(1, T-1):
            if (t == 1) & (self._AUXILIARY_CELL_COST is not None):
                M_list.append( K[t] * ( np.outer(u[t][:-1]*s, np.vstack((v[t+1],v[T])) ) ) )
            else:
                M_list.append( K[t] * ( np.outer(u[t]*s, np.vstack((v[t+1],v[T])) ) ) )

        M_list.append( K[T-1] * np.outer(u[T-1]*s, v[T]) )

        return M_list


    def _compute_feasibility(self):
        """计算边际约束的违背程度

        该函数评估当前传输计划是否满足多阶段最优传输的边际约束条件。
        违背程度（infeasibility）用于判断Sinkhorn算法是否收敛。

        边际约束条件包括：
        1. 初始边际约束 (delta_0): 初始细胞发出的总质量应等于1
        2. 中间边际约束 (delta_n): 每个中间细胞发出的总质量应等于其接收的总质量
        3. 终端边际约束 (delta_T): 终端细胞接收的总质量应等于1
        4. 时间连续性约束 (delta_t): 相邻时间步骤的质量守恒

        具体计算：
        - delta_0 = 1 - min(sum(M_0, axis=1))：初始细胞行和的最小值偏离1的程度
        - delta_n = 1 - min(sum(M[1:-1], axis=1) + sum(M_T, axis=1))：中间细胞质量平衡
        - delta_T = 1 - min(sum(M[:, -nF:], axis=0))：终端细胞列和的最小值偏离1的程度
        - delta_t = max(abs(sum(M_{t-1}[:, :n], axis=0) - sum(M_t, axis=1)))：相邻阶段质量差异

        Returns:
            无返回值，结果存储在以下实例属性中：
            - self.delta_dict: 字典，包含各约束的最大违背值
            - self.delta_vec: numpy数组，包含所有delta值
        """
        T = self._T

        M = 0
        for i in range(1,len(self.transport_plans)-1):
            M += self.transport_plans[i]

        if (self._AUXILIARY_CELL_COST is not None):
            M = M[:-1,:]

            delta_0 = 1 - np.minimum(np.min(np.sum(self.transport_plans[0][:-1], axis=1)), 1)
            delta_n = 1 - np.minimum(np.min(np.sum(M,axis=1) + np.sum(self.transport_plans[-1][:-1,:],axis=1)), 1 ) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[1],axis=1)), 1)
            delta_T = 1 - np.minimum(np.min(np.sum([np.sum(self.transport_plans[k][:,-self._nF:-1],axis=0) for k in range(1,len(self.transport_plans))],axis=0)), 1) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[-1],axis=0)), 1)

        else:
            delta_0 = 1 - np.minimum(np.min(np.sum(self.transport_plans[0], axis=1)), 1)
            delta_n = 1 - np.minimum(np.min(np.sum(M,axis=1) + np.sum(self.transport_plans[-1],axis=1)), 1 ) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[1],axis=1)), 1)
            delta_T = 1 - np.minimum(np.min(np.sum([np.sum(self.transport_plans[k][:,-self._nF:],axis=0) for k in range(1,len(self.transport_plans))],axis=0)), 1) if (self._NUM_GROUPS > 1) else 1 - np.minimum(np.min(np.sum(self.transport_plans[-1],axis=0)), 1)

        delta_t = []
        for t in range(1,T):
            delta_t.append( np.max(np.abs(np.sum(self.transport_plans[t-1][:,:self._n], axis=0) - np.sum(self.transport_plans[t], axis=1))) )

        delta  = np.array([delta_0] + [delta_n] + delta_t + [delta_T])

        self.delta_dict = {'mu_0' : delta_0, 'mu_sum' : delta_n, 'max_mu_t' : np.max(delta_t), 'mu_T' : delta_T}

        self.delta_vec = delta

        return




    def _printouts(self, iter, max_step, infeasibility, curr_time ):
        """打印Sinkhorn迭代进度信息

        在每次迭代（当 iteration % patience == 0 时）打印当前优化状态，
        包括迭代次数、对偶变量更新步长、边际约束违背程度和已用时间。

        输出格式示例：
        Iteration: 100 [========] Max dual step: 1.234e-05 | Infeasibility: 5.678e-06 | Elapsed time: 0:01:23

        Args:
            iter: 当前迭代次数
            max_step: 对偶变量更新的最大步长（用于判断收敛）
            infeasibility: 边际约束违背程度的最大值
            curr_time: 从迭代开始到现在经过的时间（秒）

        Returns:
            无返回值，直接输出到标准输出
        """
        print("\r", "Iteration: {k} [========]".format(k=iter) + " Max dual step: {0:.3e}".format(max_step) + " | Infeasibility: {0:.3e}".format(infeasibility) + " | Elapsed time: {time}".format(time=timedelta(seconds=curr_time)), end = "", flush=True)

        return


    def _sinkhorn_iterations(self):
        """执行Sinkhorn迭代求解多阶段最优传输问题

        这是MultistageOT的核心算法，使用块坐标上升法(Block-Coordinate Ascent)求解
        熵正则化最优传输问题的对偶问题。

        算法背景:
        多阶段最优传输将细胞分化过程建模为T个阶段的传输问题。在每个阶段t，质量从
        细胞群体t传输到细胞群体t+1，同时满足边际约束。熵正则化通过在目标函数中添加
        KL散度项来平滑问题，使得Sinkhorn算法可以高效求解。

        算法原理:
        1. 初始化对偶变量u（行缩放因子）和v（列缩放因子），以及中间缩放因子s
        2. 交替更新三个变量（块坐标上升）:
           - v[t] = 1/u[t]  （列变量更新）
           - s = max(1, 1/s_sum)  （缩放因子更新，确保质量守恒）
           - u[t] = sqrt((K^T * (s * u[t-1])) / (s * (K * v[t+1] + R)))  （行变量更新）
        3. 收敛条件: max(对偶变量更新步长, 边际约束违背程度) <= tolerance

        关键数学公式:
        对于中间阶段 t = 1, ..., T-1:
            u[t] = sqrt( (K_tilde^T @ (s * u[t-1])) / (s * (K_tilde @ v[t+1] + R)) )
            其中 R = K_hat @ v[T] 代表流向终末细胞的质量

        对于初始阶段 t = 0:
            u[0] = max(1, 1 / (K_0 @ v[1]))

        对于终末阶段 t = T:
            u[T] = min(1, K_hat^T @ (s * u_sum))
            其中 u_sum = sum(u[1:T-1]) 是所有中间阶段的行缩放因子之和

        缩放因子 s:
            s = max(1, 1 / (sum_t u[t] * (K_tilde @ v[t+1]) + u_sum * R))
            s >= 1 确保传输计划的质量不超过边际约束

        收敛判断:
        - max_step: 对偶变量在对数空间中的最大变化量（衡量优化进度）
        - infeasibility: 边际约束的最大违背程度（衡量解的可行性）
        - 当两者都小于 tolerance 时收敛

        Returns:
            None（结果存储在以下实例属性中）:
            - self.transport_plans: 传输计划列表
            - self.dual_variables: 对偶变量列表
            - self.utility_variables: 工具变量（u, v, s）
            - self.history: 迭代历史记录
        """
        T = self._T
        self.__C = self._get_C()

        # =====================================================================
        # 第一步：计算K矩阵（Gibbs核矩阵）
        # K = exp(-C/epsilon)，其中C是代价矩阵，epsilon是正则化参数
        # K矩阵的元素K_ij表示从源i到目标j的"亲和度"
        # =====================================================================
        if self._SPARSE: #稀疏实现（当矩阵很大时节省内存）
            K             = scipy.sparse.csr_array(np.exp(-self.__C[1] / self._EPSILON))
            print("Sparsity of K: ", scipy.sparse.csr_matrix.count_nonzero(K) / (K.shape[0]*K.shape[1]))
        else:
            K         = np.exp(-self.__C[1] / self._EPSILON)

        # K_0: 初始阶段到第一阶段的核矩阵 (n0 x (n + nF))
        K_0           = np.exp(-self.__C[0] / self._EPSILON)
        # K_0_tilde: K_0的前n列（对应中间细胞，排除终末细胞）
        K_0_tilde     = K_0[:,:self._n]
        # K_hat: 中间/最后阶段到终末细胞的核矩阵
        K_hat         = np.exp(-self.__C[2] / self._EPSILON)
        # K_tilde: 中间阶段之间的核矩阵（排除终末细胞列）
        K_tilde       = K[:,:self._n]

        # =====================================================================
        # 计算转置矩阵（用于后续更新公式中的矩阵乘法）
        # =====================================================================
        K_0_trans       = K_0.T
        K_0_tilde_trans = K_0_tilde.T
        K_hat_trans     = K_hat.T
        K_tilde_trans   = K_tilde.T

        # =====================================================================
        # 第二步：初始化对偶变量
        # u: 行缩放因子列表，u[t] 是阶段t的行缩放因子
        # v: 列缩放因子列表，v[t] = 1/u[t]（Sinkhorn性质）
        # s: 中间缩放因子，用于平衡不同阶段之间的质量流
        # =====================================================================
        if len(self.dual_variables) > 0: #如果模型已经训练过，从当前估计开始（热启动）
            u = [np.exp(self.dual_variables[t]/self._EPSILON) for t in range(T + 1)]
            s = np.exp(self.dual_variables[-1]/self._EPSILON)
            v = [1 / u[t] for t in range(T + 1)]
        else:    #否则，从初始猜测开始
            u     = [np.ones((self._n,1)) for t in range(T + 1)]
            if self._AUXILIARY_CELL_COST is not None:
                u[1] = np.ones((self._n+1,1))

            u[0]  = np.ones((self._n0,1))
            u[T]  = np.ones((self._nF,1))

            v = [1 / u[t] for t in range(T + 1)]

            s = np.ones((self._n,1))

        # 计算u_sum：所有中间阶段行缩放因子的和
        # 这个量在更新u[T]时需要用到
        if (self._AUXILIARY_CELL_COST is not None):
                u_sum    = np.sum(u[2:-1],axis=0) + u[1][:-1]
        else:
            u_sum    = np.sum(u[1:-1],axis=0)

        # 初始化历史记录列表
        if self.history is None:
            max_steps       = []
            infeasibilities = []
            ts              = []
        else:
            max_steps       = self.history['max_steps']
            infeasibilities = self.history['infeasibility']
            ts              = self.history['wall_clock_time']

        iteration = 0

        # =====================================================================
        # 计算初始传输计划和可行性（用于收敛判断）
        # =====================================================================
        self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        self._compute_feasibility()
        infeasibility = np.max(self.delta_vec)
        max_step   = 1

        # =====================================================================
        # 第三步：Sinkhorn迭代主循环
        # 块坐标上升法交替更新 v -> s -> u
        # =====================================================================
        start_time = time.time()
        while not self._converged(infeasibility, max_step):

            # 保存旧的对偶变量（用于计算更新步长）
            old_u = copy.deepcopy(u)
            old_s = copy.deepcopy(s)

            # =================================================================
            # 更新 v（列缩放因子）
            # Sinkhorn性质: v[t] = 1/u[t]
            # 这确保了传输计划的列和等于目标边际
            # =================================================================
            v = [1 / u[t] for t in range(T + 1)]
            # R: 从中间阶段流向终末细胞的总质量
            # R = K_hat @ v[T]，用于更新中间阶段的u和s
            R =  K_hat @ v[T]

            # =================================================================
            # 更新 s（中间缩放因子）
            # s确保每个中间细胞发出的质量等于其接收的质量（质量守恒）
            # s_sum = sum_t u[t] * (K_tilde @ v[t+1]) + u_sum * R
            # s = max(1, 1/s_sum)  （s >= 1 约束）
            # =================================================================
            if (self._AUXILIARY_CELL_COST is not None):
                s_sum = u[1][:-1] * (K_tilde @ v[2]) + np.sum([u[t] * (K_tilde @ v[t + 1]) for t in range(2, T-1)], axis=0) +  u_sum*R
            else:
                s_sum = np.sum([u[t] * (K_tilde @ v[t + 1]) for t in range(1, T-1)], axis=0) +  u_sum*R
            s = np.maximum(self._1n, 1 / s_sum)

            if (self._AUXILIARY_CELL_COST is not None):
                s[-1,0] = 1

            # =================================================================
            # 更新 u（行缩放因子）
            # u[0]: 初始阶段，确保从初始细胞发出的质量等于1
            # u[1:T-1]: 中间阶段，使用几何平均公式更新
            # u[T]: 终末阶段，确保流入终末细胞的质量等于1
            # =================================================================

            # 更新 u[0]（初始阶段）: u[0] = max(1, 1/(K_0 @ v[1]))
            u[0] = np.maximum(self._1n0, 1 / (K_0 @ v[1]) )
            if (self._AUXILIARY_CELL_COST is not None):
                u[0][-1,0] = 1

            # 更新 u[1]（第一个中间阶段）
            # u[1] = sqrt( (K_0_tilde^T @ u[0]) / (s * (K_tilde @ v[2] + R)) )
            if (self._AUXILIARY_CELL_COST is not None):
                u[1][:-1] = np.sqrt((K_0_tilde_trans @ u[0]) / (s * (K_tilde @ v[2] + R)))
                u[1][-1,0] = 1
            else:
                u[1] = np.sqrt((K_0_trans @ u[0]) / (s * (K_tilde @ v[2] + R)))

            # 更新 u[2:T-1]（中间阶段）
            # u[t] = sqrt( (K_tilde^T @ (s * u[t-1])) / (s * (K_tilde @ v[t+1] + R)) )
            for t in range(2, T-1):
                if (t == 2) & (self._AUXILIARY_CELL_COST is not None):
                    u[t] = np.sqrt((K_tilde_trans @ (s * u[t-1][:-1])) / (s * (K_tilde @ v[t+1] + R)))
                else:
                    u[t] = np.sqrt((K_tilde_trans @ (s * u[t-1])) / (s * (K_tilde @ v[t+1] + R)))

            # 更新 u[T-1]（最后一个中间阶段）
            # u[T-1] = sqrt( (K_tilde^T @ (s * u[T-2])) / (s * (K_hat @ v[T])) )
            u[T-1] = np.sqrt((K_tilde_trans @ (s * u[T-2])) / (s * (K_hat @ v[T])))

            # 重新计算u_sum（用于更新u[T]）
            if (self._AUXILIARY_CELL_COST is not None):
                u_sum    = np.sum(u[2:-1],axis=0) + u[1][:-1]
            else:
                u_sum    = np.sum(u[1:-1],axis=0)

            # 更新 u[T]（终末阶段）: u[T] = min(1, K_hat^T @ (s * u_sum))
            u[T]     = np.minimum(self._1nT, K_hat_trans @ (s * u_sum))

            if (self._AUXILIARY_CELL_COST is not None):
                u[T][-1,0] = 1

            # =================================================================
            # 定期检查收敛性（每PATIENCE次迭代）
            # =================================================================
            if (iteration%self._PATIENCE == 0):

                self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)
                self._compute_feasibility()

                # 计算对偶变量的最大更新步长
                # 在对数空间中计算: step = |epsilon * log(old_u) - epsilon * log(new_u)|
                max_step = np.maximum(np.max([np.max(np.abs(self._EPSILON*np.log(old_u[t]) - self._EPSILON*np.log(u[t])))  for t in range(len(old_u))]), np.max(np.abs(self._EPSILON*np.log(old_s) - self._EPSILON*np.log(s))))

                max_steps.append(max_step)


                infeasibility = np.max(self.delta_vec)

                infeasibilities.append(infeasibility)

                curr_time = time.time() - start_time
                if self._VERBOSE:

                    self._printouts(iteration, max_step, infeasibility, curr_time)
                ts.append(curr_time)

            # =================================================================
            # 保存检查点（用于长时间运行的恢复）
            # =================================================================
            if (self._path_to_checkpoints is not None) and (iteration%self._checkpoints== 0):

                self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

                self.dual_variables = [] #Reset dual variables
                # 将对偶变量转换回对数空间存储
                for t in range(len(u)):
                    self.dual_variables.append(self._EPSILON*np.log(u[t]))
                self.dual_variables.append(self._EPSILON*np.log(s))

                self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}


                #Create directory:
                directory = self._path_to_checkpoints + "/checkpoint_{it}".format(it=iteration)
                os.mkdir(directory)

                self.save_model(directory)


            del old_s
            del old_u

            iteration += 1

        # =====================================================================
        # 第四步：收敛后处理
        # =====================================================================
        if self._VERBOSE:
            print("\n")
            print("Sinkhorn algorithm converged to a solution within the given tolerance ({0:.4e}) of both feasibility and max dual-variable update step.".format(self._TOLERANCE))
            # Retrieve transport plans:
            print("\n")
            print("Retrieving transport plans...")

        self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        if self._VERBOSE:
            print("Done.")

        # 存储对偶变量（转换回对数空间）
        self.dual_variables = [] #Reset dual variables
        if self._VERBOSE:
            print("Storing dual variables...")
        for t in range(len(u)):
            self.dual_variables.append(self._EPSILON*np.log(u[t]))
        self.dual_variables.append(self._EPSILON*np.log(s))
        if self._VERBOSE:
            print("Done.")

        # 存储工具变量（用于热启动或后续分析）
        for t in range(len(u)):
            self.utility_variables[0].append(u[t])
        for t in range(len(v)):
            self.utility_variables[1].append(v[t])
        self.utility_variables[2] = s

        if self._VERBOSE:
            print("Max absolute feasibility errors in the marginal constraints: \n ", self.delta_dict)
        self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}

        return


    def _sinkhorn_iterations_with_prior(self):
        """执行带近端先验的Sinkhorn迭代求解多阶段最优传输问题

        这是Sinkhorn算法的变体，使用先前获得的传输计划作为"先验"(prior)来加速收敛。
        通过近端正则化(proximal regularization)，算法可以从前一次优化的结果热启动，
        从而更快地收敛到新的正则化参数下的最优解。

        算法背景:
        在proximal_sinkhorn方法中，我们需要逐步减小epsilon来逼近无正则化解。
        直接从头开始求解每个epsilon下的问题会非常耗时。通过使用前一个epsilon下的
        传输计划作为先验，我们可以显著加速收敛。

        与标准Sinkhorn的区别:
        1. 使用G矩阵代替K矩阵: G_t = K_t * P_t，其中P_t是先验传输计划
        2. 更新公式中增加了先验项，使得解倾向于接近先验
        3. 包含数值稳定性检查：当检测到NaN或零值时，自动增大proximal epsilon

        关键数学公式:
        G矩阵定义:
            G_t = exp(-C_t / epsilon_proximal) * P_t
            其中P_t是先验传输计划，epsilon_proximal是近端正则化参数

        缩放因子s的更新:
            s = max(1, 1 / (sum_t u[t] * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T]) + ...))
            相比标准Sinkhorn，增加了G_hat[t] @ v[T]项（来自先验的贡献）

        行缩放因子u的更新:
            u[t] = sqrt( (G_tilde[t-1]^T @ (s * u[t-1])) / (s * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T])) )

        数值稳定性:
        当检测到u中出现NaN或零值时:
        1. 将proximal epsilon增大10%
        2. 重新计算G矩阵
        3. 重新初始化对偶变量
        4. 重置迭代计数器

        Returns:
            None（结果存储在以下实例属性中）:
            - self.transport_plans: 传输计划列表
            - self.dual_variables: 对偶变量列表
            - self.utility_variables: 工具变量（u, v, s）
            - self.history: 迭代历史记录
            - self._EPSILON: 更新后的有效正则化参数（1/epsilon = 1/epsilon_old + 1/epsilon_proximal）
        """
        T = self._T
        self.__C = self._get_C()

        # =====================================================================
        # 第一步：计算G矩阵（先验加权的核矩阵）
        # G = K * P，其中K是标准核矩阵，P是先验传输计划
        # 使用proximal_epsilon作为正则化参数（而不是标准的epsilon）
        # =====================================================================
        G_0           = np.exp(-self.__C[0] / self._PROXIMAL_EPSILON) * self.__PRIOR[0]

        G_0_tilde     = G_0[:,:self._n]
        G             = [G_0] + [np.exp(-self.__C[1] / self._PROXIMAL_EPSILON) * self.__PRIOR[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / self._PROXIMAL_EPSILON) * self.__PRIOR[T-1]]
        # G_hat: G矩阵中对应终末细胞的部分
        G_hat         = [None] + [G[t][:,self._n:] for t in range(1,T-1)] + [G[T-1]]
        # G_tilde: G矩阵中对应中间细胞的部分（排除终末细胞列）
        G_tilde       = [None] + [G[t][:,:self._n] for t in range(1,T-1)]

        # =====================================================================
        # 计算转置矩阵
        # =====================================================================
        G_0_trans     = G_0.T
        G_0_tilde_trans = G_0_tilde.T
        G_hat_trans   = [None] + [G_hat[t].T for t in range(1,T)]
        G_tilde_trans = [None] + [G_tilde[t].T for t in range(1,T-1)]



        # =====================================================================
        # 第二步：初始化对偶变量
        # 如果已有训练好的工具变量，使用它们进行热启动
        # =====================================================================
        if len(self.utility_variables[0]) > 0: #如果模型已经训练过，从当前估计开始
            u = [self.utility_variables[0][t] for t in range(T + 1)]
            v = [self.utility_variables[1][t] for t in range(T + 1)]
            s = self.utility_variables[2]


        else:    #否则，从初始猜测开始（可能导致不稳定）
            u     = [np.ones((self._n,1)) for t in range(T + 1)]
            if self._AUXILIARY_CELL_COST is not None:
                u[1] = np.ones((self._n+1,1))
            u[0]  = np.ones((self._n0,1))
            u[T]  = np.ones((self._nF,1))
            v = [1 / u[t] for t in range(T + 1)]
            s = np.ones((self._n,1))

        # 初始化历史记录
        if self.history is None:
            max_steps       = []
            infeasibilities = []
            ts              = []
        else:
            max_steps       = self.history['max_steps']
            infeasibilities = self.history['infeasibility']
            ts              = self.history['wall_clock_time']

        iteration = 0

        # =====================================================================
        # 计算初始传输计划和可行性
        # =====================================================================
        self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
        self._compute_feasibility()
        infeasibility = np.max(self.delta_vec)
        max_step   = 1


        # =====================================================================
        # 第三步：Sinkhorn迭代主循环
        # =====================================================================
        start_time = time.time()



        while not self._converged(infeasibility, max_step):

            # =================================================================
            # 数值稳定性检查
            # 检测u中是否出现NaN或零值，这些会导致算法失败
            # =================================================================
            u_list = []
            for elem in u:
                u_list += elem.flatten().tolist()
            nan_detected_in_u = np.isnan(u_list).any()
            zeros_detected_in_u = (np.array(u_list).size - np.count_nonzero(u_list)) > 0


            #############################################################################################
            # 如果检测到不稳定性，增大proximal epsilon并重新开始
            #############################################################################################
            if nan_detected_in_u or zeros_detected_in_u:# np.isnan(max_step):


                #增大proximal epsilon以提高稳定性
                self._PROXIMAL_EPSILON = 1.1*self._PROXIMAL_EPSILON

                if self._VERBOSE:
                    print("NaNs encountered, increasing proximal epsilon to ->", self._PROXIMAL_EPSILON)

                # =================================================================
                # 重新计算G矩阵（使用新的proximal epsilon）
                # =================================================================
                G_0           = np.exp(-self.__C[0] / self._PROXIMAL_EPSILON) * self.__PRIOR[0]

                G_0_tilde     = G_0[:,:self._n]


                G             = [G_0] + [np.exp(-self.__C[1] / self._PROXIMAL_EPSILON) * self.__PRIOR[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / self._PROXIMAL_EPSILON) * self.__PRIOR[T-1]]
                G_hat         = [None] + [G[t][:,self._n:] for t in range(1,T-1)] + [G[T-1]]
                G_tilde       = [None] + [G[t][:,:self._n] for t in range(1,T-1)]

                #重新计算转置矩阵
                G_0_trans     = G_0.T
                G_0_tilde_trans = G_0_tilde.T
                G_hat_trans   = [None] + [G_hat[t].T for t in range(1,T)]
                G_tilde_trans = [None] + [G_tilde[t].T for t in range(1,T-1)]


                # =================================================================
                # 重新初始化对偶变量
                # =================================================================
                if (len(self.utility_variables[0]) > 0) and (iteration > 0): #如果模型已训练且不是第一次迭代，从当前估计开始
                    u = [self.utility_variables[0][t] for t in range(T + 1)]
                    v = [self.utility_variables[1][t] for t in range(T + 1)]
                    s = self.utility_variables[2]

                else:   #否则，从不同的初始猜测开始
                    u     = [np.ones((self._n,1)) for t in range(T + 1)]
                    if self._AUXILIARY_CELL_COST is not None:
                        u[1] = np.ones((self._n+1,1))
                    u[0]  = np.ones((self._n0,1))
                    u[T]  = np.ones((self._nF,1))
                    v = [1 / u[t] for t in range(T + 1)]
                    s = np.ones((self._n,1))

                if self.history is None:
                    max_steps       = []
                    infeasibilities = []
                    ts              = []
                else:
                    max_steps       = self.history['max_steps']
                    infeasibilities = self.history['infeasibility']
                    ts              = self.history['wall_clock_time']

                iteration = 0
            #############################################################################################


            # 保存旧的对偶变量
            old_u = copy.deepcopy(u)
            old_s = copy.deepcopy(s)


            # =================================================================
            # 更新 v（列缩放因子）: v[t] = 1/u[t]
            # =================================================================
            v = [1 / u[t] for t in range(T + 1)]


            # =================================================================
            # 更新 s（中间缩放因子）
            # 与标准Sinkhorn不同，这里包含G_hat[t] @ v[T]项（先验贡献）
            # s = max(1, 1 / (u[T-1] * (G_hat[T-1] @ v[T]) + sum_t u[t] * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T])))
            # =================================================================
            if (self._AUXILIARY_CELL_COST is not None):
                s = np.maximum(self._1n, 1 / ( u[T-1] * (G_hat[T-1] @ v[T]) + u[1][:-1]*(G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) + np.sum([u[t] * (G_tilde[t] @ v[t + 1] + G_hat[t] @ v[T] ) for t in range(2, T-1)], axis=0) ) )
            else:
                s = np.maximum(self._1n, 1 / ( u[T-1] * (G_hat[T-1] @ v[T]) + np.sum([u[t] * (G_tilde[t] @ v[t + 1] + G_hat[t] @ v[T] ) for t in range(1, T-1)], axis=0) ) )

            if (self._AUXILIARY_CELL_COST is not None):
                s[-1,0] = 1

            # =================================================================
            # 更新 u（行缩放因子）
            # =================================================================

            # 更新 u[0]（初始阶段）
            u[0] = np.maximum(self._1n0, 1 / (G_0 @ v[1]) )

            if (self._AUXILIARY_CELL_COST is not None):
                u[0][-1,0] = 1

            # 更新 u[1]（第一个中间阶段）
            # u[1] = sqrt( (G_0_tilde^T @ u[0]) / (s * (G_tilde[1] @ v[2] + G_hat[1] @ v[T])) )
            if (self._AUXILIARY_CELL_COST is not None):
                u[1][:-1] = np.sqrt((G_0_tilde_trans @ u[0]) / (s * ( G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) ))
                u[1][-1,0] = 1
            else:
                u[1] = np.sqrt((G_0_trans @ u[0]) / (s * ( G_tilde[1] @ v[2] + G_hat[1] @ v[T] ) ))

            # 更新 u[2:T-1]（中间阶段）
            for t in range(2, T-1):
                if (t == 2) & (self._AUXILIARY_CELL_COST is not None):
                    u[t] = np.sqrt((G_tilde_trans[t-1] @ (s * u[t-1][:-1])) / (s * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T]) ))
                else:
                    u[t] = np.sqrt((G_tilde_trans[t-1] @ (s * u[t-1])) / (s * (G_tilde[t] @ v[t+1] + G_hat[t] @ v[T]) ))

            # 更新 u[T-1]（最后一个中间阶段）
            u[T-1] = np.sqrt((G_tilde_trans[T-2] @ (s * u[T-2])) / (s * (G_hat[T-1] @ v[T]) ))

            # 更新 u[T]（终末阶段）
            # u[T] = min(1, G_hat[T-1]^T @ (s * u[T-1]) + sum_t G_hat[t]^T @ (s * u[t]))
            if (self._AUXILIARY_CELL_COST is not None):
                u[T]   = np.minimum(self._1nT, G_hat_trans[T-1]@(s*u[T-1]) + G_hat_trans[1] @ (s*u[1][:-1])  + np.sum([G_hat_trans[t] @ (s*u[t]) for t in range(2, T-1)], axis=0) )
            else:
                u[T]   = np.minimum(self._1nT, G_hat_trans[T-1]@(s*u[T-1]) + np.sum([G_hat_trans[t] @ (s*u[t]) for t in range(1, T-1)], axis=0) )

            if (self._AUXILIARY_CELL_COST is not None):
                u[T][-1,0] = 1


            # =================================================================
            # 计算对偶变量的最大更新步长
            # =================================================================
            max_step = np.maximum(np.max([np.max(np.abs(self._PROXIMAL_EPSILON*np.log(old_u[t]) - self._PROXIMAL_EPSILON*np.log(u[t])))  for t in range(len(old_u))]), np.max(np.abs(self._PROXIMAL_EPSILON*np.log(old_s) - self._PROXIMAL_EPSILON*np.log(s))))



            # =================================================================
            # 定期检查收敛性
            # =================================================================
            if (iteration%self._PATIENCE == 0):
                self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
                self._compute_feasibility()



                curr_time = time.time() - start_time
                max_steps.append(max_step)
                ts.append(curr_time)

                infeasibility = np.max(self.delta_vec)

                infeasibilities.append(infeasibility)

                if self._VERBOSE:
                    self._printouts(iteration, max_step, infeasibility, curr_time)

            # =================================================================
            # 保存检查点
            # =================================================================
            if (self._path_to_checkpoints is not None) and (iteration%self._checkpoints== 0):

                self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)


                # Store dual variables:
                for t in range(len(u)):
                    self.utility_variables[0].append(u[t])
                for t in range(len(v)):
                    self.utility_variables[1].append(v[t])
                self.utility_variables[2] = s

                self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}


                #Create directory:
                directory = self._path_to_checkpoints + "/checkpoint_{it}".format(it=iteration)
                os.mkdir(directory)

                self.save_model(directory)


            del old_s
            del old_u

            iteration += 1


        # =====================================================================
        # 第四步：收敛后处理
        # =====================================================================

        #释放内存：删除先验传输计划
        del self.__PRIOR

        if self._VERBOSE:
            print("\n")
            print("Sinkhorn algorithm converged to a solution within the given tolerance ({0:.4e}) of both feasibility and max dual-variable update step.".format(self._TOLERANCE))
            # Retrieve transport plans:
            print("\n")
            print("Retrieving transport plans...")

        self.transport_plans = self._retrieve_plans_proximal(G_0, G, u, v, s)
        if self._VERBOSE:
            print("Done.")

        if self._VERBOSE:
            print("Storing utility variables...")


        # 存储对偶变量（转换回对数空间，使用proximal_epsilon作为缩放因子）
        self.dual_variables = [] #Reset dual variables
        if self._VERBOSE:
            print("Storing dual variables...")
        for t in range(len(u)):
            self.dual_variables.append(self._PROXIMAL_EPSILON*np.log(u[t]))
        self.dual_variables.append(self._PROXIMAL_EPSILON*np.log(s))
        if self._VERBOSE:
            print("Done.")


        # 存储工具变量（用于后续的proximal迭代）
        self.utility_variables      = [[],[],0] #Reset utility_variables

        for t in range(len(u)):
            self.utility_variables[0].append(u[t])
        for t in range(len(v)):
            self.utility_variables[1].append(v[t])

        self.utility_variables[2] = s

        if self._VERBOSE:
            print("Done.")

        if self._VERBOSE:
            print("Max absolute feasibility errors in the marginal constraints: \n ", self.delta_dict)
        self.history = {'max_steps' : max_steps, 'wall_clock_time' : ts, 'infeasibility' : infeasibilities}

        # =====================================================================
        # 更新有效正则化参数
        # 新的epsilon满足: 1/epsilon = 1/epsilon_old + 1/epsilon_proximal
        # 这确保了proximal迭代的收敛性
        # =====================================================================
        self._EPSILON = 1/(1/self._EPSILON + 1/self._PROXIMAL_EPSILON)
        return        


    #Public methods:
    def save_model(self, path):
        """将训练好的多阶段最优传输模型保存到指定路径

        该方法将模型的所有状态持久化到磁盘，包括：
        - 对偶变量（lambda_0, lambda_1, ..., lambda_T, rho）：Sinkhorn迭代的优化结果
        - 先验工具变量（仅proximal模式）：用于热启动近端Sinkhorn迭代
        - 细胞分类（initial_cells, intermediate_cells, terminal_cells）：细胞群体划分
        - 原始数据：拟合模型所用的基因表达数据
        - 模型超参数（NUM_GROUPS, EPSILON, PATIENCE, TOLERANCE, AUXILIARY_CELL_COST）
        - proximal相关参数（PROXIMAL_EPSILON_HISTORY, TOTAL_EPSILON_HISTORY, INNER_ITERATIONS）
        - 收敛历史（wall_clock_time, max_steps, infeasibility）

        保存格式说明：
        - 对偶变量和参数：.npy格式（NumPy二进制格式，高效存储数组）
        - 原始数据：.csv格式（便于跨平台读取）

        注意事项：
        - 目标目录必须为空，以避免意外覆盖已有文件
        - 文件命名规则：lambda_{t}.npy（对偶变量）、rho.npy（缩放因子）、
          {i}_prior_u_{t}.npy（先验行缩放因子）等

        Args:
            path (str): 保存目录路径，必须以 '/' 结尾，目录必须为空

        Returns:
            None
        """
        # 安全检查：确保目标目录为空，防止覆盖已有模型文件
        if (len(os.listdir(path)) != 0):
            sys.exit('WARNING: The given path directory is NOT empty. Please specify an empty directory to avoid overwriting an existing model. ')

        print(">>> Saving MMOT model...")
        print("To: ", path)

        # 保存对偶变量 lambda_0, lambda_1, ..., lambda_{T-1}
        # 每个lambda_t是一个向量，存储为独立的.npy文件
        for t in range(len(self.transport_plans)):
            np.save(path + "lambda_{t}".format(t=t), self.dual_variables[t])
        # 保存最后一个对偶变量 lambda_T（对应终端阶段）
        np.save(path + "lambda_{t}".format(t=t+1), self.dual_variables[t+1])
        # 保存缩放因子 rho（中间细胞的质量守恒缩放向量）
        np.save(path + "rho", self.dual_variables[-1])

        # 如果使用了proximal方案，保存所有先验工具变量
        # 每次proximal迭代的u, v, s都需要保存，用于加载时重建传输计划
        if self.__USE_PRIOR:
            for i in range(len(self.__prior_utility_variables)):
                for t in range(self._T+1):
                    # 保存第i次迭代、第t阶段的行缩放因子u和列缩放因子v
                    np.save(path + "{i}_prior_u_{t}".format(t=t,i=i), self.__prior_utility_variables[i][0][t])
                    np.save(path + "{i}_prior_v_{t}".format(t=t,i=i), self.__prior_utility_variables[i][1][t])
                # 保存第i次迭代的中间缩放因子s
                np.save(path + "{i}_prior_s".format(i=i), self.__prior_utility_variables[i][2])

        # 保存细胞分类信息
        np.save(path + "initial_cells", self.initial_cells)        # 初始细胞（干细胞）
        np.save(path + "intermediate_cells", self.intermediate_cells)  # 中间细胞（过渡态）
        np.save(path + "terminal_cells", self.terminal_cells)      # 终端细胞（终末分化）

        # 保存原始数据（基因表达矩阵）为CSV格式
        self.data.to_csv(path + "data.csv")

        # 保存模型超参数
        np.save(path + "NUM_GROUPS", self._NUM_GROUPS)             # 中间阶段数

        # 保存proximal方案相关的迭代历史（仅proximal模式）
        if self.__USE_PRIOR:
            np.save(path +"PROXIMAL_EPSILON_HISTORY", self._PROXIMAL_EPSILON_HISTORY)  # proximal epsilon变化记录
            np.save(path +"TOTAL_EPSILON_HISTORY", self._TOTAL_EPSILON_HISTORY)        # 总epsilon变化记录
            np.save(path +"INNER_ITERATIONS", self._INNER_ITERATIONS)                  # 每次外层迭代的内层迭代数
            np.save(path + "OUTER_PROXIMAL_ITERATIONS", self._OUTER_PROXIMAL_ITERATIONS)  # 外层迭代总次数

        # 保存核心模型参数
        np.save(path + "AUXILIARY_CELL_COST", self._AUXILIARY_CELL_COST)  # 辅助细胞成本（离群点吸收）
        np.save(path + "EPSILON", self._EPSILON)                          # 熵正则化参数
        np.save(path + "PATIENCE", self._PATIENCE)                        # 收敛检查频率
        np.save(path + "TOLERANCE", self._TOLERANCE)                      # 收敛容忍度

        # 保存收敛历史记录
        np.save(path + "WALL_CLOCK_TIME", self.history['wall_clock_time'])  # 每次检查点的墙钟时间
        np.save(path + "MAX_STEPS", self.history['max_steps'])              # 每次检查点的最大对偶步长
        np.save(path + "INFEASIBILITY", self.history['infeasibility'])      # 每次检查点的不可行性

        print("Done.")
        return
    

    def load_model(self, path):
        """从指定路径加载已保存的多阶段最优传输模型

        该方法从磁盘恢复模型的完整状态，包括：
        - 对偶变量（lambda_0, lambda_1, ..., lambda_T, rho）：恢复优化变量
        - 先验工具变量（仅proximal模式）：恢复近端Sinkhorn的迭代历史
        - 细胞分类：恢复初始、中间、终末细胞的索引
        - 原始数据：恢复基因表达矩阵
        - 模型超参数：恢复NUM_GROUPS, EPSILON, PATIENCE, TOLERANCE等
        - 收敛历史：恢复wall_clock_time, max_steps, infeasibility

        加载流程概述：
        1. 读取模型参数文件（.npy格式）
        2. 读取原始数据文件（.csv格式）
        3. 根据是否使用proximal方案，选择不同的传输计划恢复策略：
           - 标准方案：直接从对偶变量通过公式 M = K * (u * v') 计算传输计划
           - proximal方案：需要按顺序重建所有proximal迭代的传输计划链
        4. 恢复工具变量（u, v, s）用于后续热启动

        Args:
            path (str): 模型文件所在目录路径，必须以 '/' 结尾，
                       目录应包含 save_model() 生成的所有文件

        Returns:
            None（模型状态直接更新到当前实例）
        """
        def get_file_names_containing_string(string):
            """辅助函数：获取目录中包含指定字符串的文件名列表"""
            full_list = os.listdir(path)
            final_list = [filename for filename in full_list if string in filename]
            return final_list

        print("<<< Loading MMOT model...")
        print("From: ", path)

        # 重置公有属性（耦合强度）
        self.couplings = None   # Strength of couplings between data points based on optimal transport plans (type: np.array).

        # 加载辅助细胞成本参数（若不存在则设为None）
        try:
            self._AUXILIARY_CELL_COST = np.load(path + "AUXILIARY_CELL_COST.npy")
        except:
            self._AUXILIARY_CELL_COST = None

        # 加载对偶变量 lambda_0, lambda_1, ..., lambda_{T}
        # lambda_files包含所有以"lambda_"开头的文件，数量对应T+1个阶段
        lambda_files = get_file_names_containing_string("lambda_")

        self.dual_variables = [np.load(path + "lambda_{t}.npy".format(t=t)) for t in range(len(lambda_files))]
        # 加载缩放因子 rho（最后一个对偶变量）
        self.dual_variables.append(np.load(path + "rho.npy"))

        # 检测是否存在先验工具变量（proximal模式的标志）
        prior_u_files = get_file_names_containing_string("prior_u_")
        prior_v_files = get_file_names_containing_string("prior_v_")


        if len(prior_u_files) > 0 or len(prior_v_files) > 0:
            self.__USE_PRIOR = True


        if self.__USE_PRIOR:
            # 加载proximal方案的元数据
            self._OUTER_PROXIMAL_ITERATIONS = int(np.load(path + "OUTER_PROXIMAL_ITERATIONS.npy"))
            self.__prior_utility_variables = []

            # 加载proximal epsilon和总epsilon的历史记录
            self._PROXIMAL_EPSILON_HISTORY = np.load(path + "PROXIMAL_EPSILON_HISTORY.npy").tolist()
            self._TOTAL_EPSILON_HISTORY = np.load(path + "TOTAL_EPSILON_HISTORY.npy").tolist()
            # 加载每次外层迭代对应的内层迭代数
            self._INNER_ITERATIONS = np.load(path + "INNER_ITERATIONS.npy").tolist()

            # 遍历每次proximal迭代，加载对应的先验工具变量(u, v, s)
            for i in range(self._OUTER_PROXIMAL_ITERATIONS+1):

                temp = [0,0,0]

                # 加载第i次迭代所有阶段的行缩放因子u和列缩放因子v
                temp[0] = [np.load(path + "{i}_prior_u_{t}.npy".format(t=t,i=i)) for t in range(len(lambda_files))]
                temp[1] = [np.load(path + "{i}_prior_v_{t}.npy".format(t=t,i=i)) for t in range(len(lambda_files))]
                # 加载第i次迭代的中间缩放因子s
                temp[2] = np.load(path + "{i}_prior_s.npy".format(i=i))

                self.__prior_utility_variables.append(temp)


        # 加载核心模型参数
        self._NUM_GROUPS = int(np.load(path + "NUM_GROUPS.npy"))  # 中间阶段数
        self._T = self._NUM_GROUPS + 1                             # 总时间点数

        T = self._T

        # 加载正则化参数epsilon
        self._EPSILON    = float(np.load(path + "EPSILON.npy"))

        # 加载细胞分类信息
        self.initial_cells         = np.load(path + "initial_cells.npy").tolist()        # 初始细胞索引
        self.intermediate_cells = np.load(path + "intermediate_cells.npy").tolist()     # 中间细胞索引
        self.terminal_cells     = np.load(path + "terminal_cells.npy").tolist()          # 终端细胞索引
        # 记录索引类型（用于后续保持数据类型一致性）
        index_type = type(self.initial_cells[0])

        # 计算各类细胞的数量（考虑辅助细胞的额外槽位）
        self._n0 = len(self.initial_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.initial_cells)
        self._n = len(self.intermediate_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.intermediate_cells)
        self._nF = len(self.terminal_cells) + 1 if (self._AUXILIARY_CELL_COST is not None) else len(self.terminal_cells)

        # 加载原始数据（基因表达矩阵）
        self.data = pd.read_csv(path + "data.csv", index_col='Unnamed: 0')
        # 恢复数据索引的原始类型
        self.data.index = self.data.index.astype(index_type)

        # 计算代价矩阵C（用于后续恢复传输计划）
        self.__C = self._get_C()


        if self.__USE_PRIOR: # 近端方案：需要按顺序重建所有proximal迭代的传输计划链
            proximal_epsilons = self._PROXIMAL_EPSILON_HISTORY[1:]

            self._PROXIMAL_EPSILON = self._PROXIMAL_EPSILON_HISTORY[-1]

            # 使用第一次proximal迭代的epsilon计算核矩阵K
            K         = np.exp(-self.__C[1] / proximal_epsilons[0])
            K_0       = np.exp(-self.__C[0] / proximal_epsilons[0])
            K_hat     = np.exp(-self.__C[2] / proximal_epsilons[0])

            # 加载第一次proximal迭代的工具变量并计算初始传输计划
            u = self.__prior_utility_variables[0][0]
            v = self.__prior_utility_variables[0][1]
            s = self.__prior_utility_variables[0][2]
            prior = self._retrieve_plans(K_0, K, K_hat, u, v, s)

            # 遍历后续proximal迭代，逐步重建传输计划链
            # 每次迭代使用前一次的传输计划作为先验
            for i in range(1,self._OUTER_PROXIMAL_ITERATIONS+1):

                # 计算近端核矩阵G = K * prior（先验加权的核矩阵）
                G_0  = np.exp(-self.__C[0] / proximal_epsilons[i]) * prior[0]
                G    = [G_0] + [np.exp(-self.__C[1] / proximal_epsilons[i]) * prior[t] for t in range(1,T-1)] + [np.exp(-self.__C[2] / proximal_epsilons[i]) * prior[T-1]]

                # 加载第i次迭代的工具变量
                u = self.__prior_utility_variables[i][0]
                v = self.__prior_utility_variables[i][1]
                s = self.__prior_utility_variables[i][2]

                # 使用proximal公式计算当前迭代的传输计划
                prior = self._retrieve_plans_proximal(G_0, G, u, v, s)

            # 最终的传输计划即为最后一次proximal迭代的结果
            self.transport_plans = prior
            del prior


        else: # 标准方案：直接从对偶变量计算传输计划
            # 计算Gibbs核矩阵
            K         = np.exp(-self.__C[1] / self._EPSILON)
            K_0       = np.exp(-self.__C[0] / self._EPSILON)
            K_hat     = np.exp(-self.__C[2] / self._EPSILON)

            # 从对偶变量恢复行缩放因子u和列缩放因子v
            u = [np.exp(self.dual_variables[t]/self._EPSILON) for t in range(T + 1)]
            s = np.exp(self.dual_variables[-1]/self._EPSILON)
            v = [1 / u[t] for t in range(T + 1)]

            # 计算传输计划：M = K * (u * v')
            self.transport_plans = self._retrieve_plans(K_0, K, K_hat, u, v, s)

        # 存储工具变量（用于后续热启动或继续proximal迭代）
        self.utility_variables = [u, v, s]

        # 加载收敛历史记录
        self.history = {'max_steps' : np.load(path + "MAX_STEPS.npy").tolist(), 'wall_clock_time' : np.load(path + "WALL_CLOCK_TIME.npy").tolist(), 'infeasibility' : np.load(path + "INFEASIBILITY.npy").tolist()}

        # 加载其他常量参数
        self._PATIENCE   = int(np.load(path + "PATIENCE.npy"))      # 收敛检查频率
        self._TOLERANCE  = float(np.load(path + "TOLERANCE.npy"))   # 收敛容忍度

        if self.__USE_PRIOR:
            # 将proximal相关历史合并到主history字典中
            self.history['proximal_sinkhorn'] = {'iterations' : self._INNER_ITERATIONS, 'proximal_epsilon' : self._PROXIMAL_EPSILON_HISTORY, 'total_epsilon' : self._TOTAL_EPSILON_HISTORY}

        print("Done.")



    def marginals(self):
        """计算边际（Marginals）

        基于拟合模型的传输计划，计算每个中间细胞在各运输阶段发送出去的质量分布。

        边际 mu_t[i] 表示细胞i在第t个阶段发送出去的总质量。
        这些边际描述了每个细胞在分化过程中的"活跃时期"——即该细胞主要在哪个阶段向外输送质量。

        对于每个中间细胞，marginals()返回其在各阶段（mu_1, mu_2, ..., mu_{T-1}）
        发送出去的质量值，帮助我们理解细胞在分化轨迹中的活跃时间窗口。

        Returns:
            pandas DataFrame:
                - 行：所有细胞的索引（包括初始细胞、中间细胞和终末细胞）
                - 列：mu_1, mu_2, ..., mu_{T-1}（每个阶段的边际）
                - 值：每个细胞在每个阶段发送的质量
                注意：初始细胞和终末细胞的边际值设为NaN，因为它们不是"中间"阶段
        """
        # 计算每个运输阶段（除去首尾阶段）各细胞发送出去的质量
        # 对每个阶段k，沿运输矩阵的行方向求和，得到每个细胞的边际值
        if self._AUXILIARY_CELL_COST is not None:
            # 存在辅助细胞时，排除最后一列（辅助相关列）
            mu_t = np.array([np.sum(self.transport_plans[k][:-1,:],axis=1) for k in range(1,len(self.transport_plans))]).T
        else:
            mu_t = np.array([np.sum(self.transport_plans[k],axis=1) for k in range(1,len(self.transport_plans))]).T

        # 生成列标签：mu_1, mu_2, ..., mu_{T-1}
        marginal_labels = ["mu_{t}".format(t=t) for t in range(1,self._T) ]

        # 创建DataFrame，行索引为中间细胞
        marginals_df = pd.DataFrame(mu_t, columns=marginal_labels, index=self.intermediate_cells)

        # 为初始细胞创建NaN占位DataFrame（初始细胞不在中间阶段发送质量）
        if self._AUXILIARY_CELL_COST is not None:
            root_df = pd.DataFrame(np.nan*np.zeros((self._n0-1, len(marginal_labels))), columns=marginal_labels, index=self.initial_cells)
        else:
            root_df = pd.DataFrame(np.nan*np.zeros((self._n0, len(marginal_labels))), columns=marginal_labels, index=self.initial_cells)

        # 为终末细胞创建NaN占位DataFrame（终末细胞不在中间阶段发送质量）
        if self._AUXILIARY_CELL_COST is not None:
            terminal_df = pd.DataFrame(np.nan*np.zeros((self._nF-1, len(marginal_labels))), columns=marginal_labels, index=self.terminal_cells)
        else:
            terminal_df = pd.DataFrame(np.nan*np.zeros((self._nF, len(marginal_labels))), columns=marginal_labels, index=self.terminal_cells)

        # 拼接所有细胞并按索引排序
        marginals_df = pd.concat((root_df, marginals_df, terminal_df)).sort_index()

        return marginals_df


    def mass_to_terminals(self):
        """计算到达终末细胞的质量（Mass to Terminals）

        基于拟合模型的传输计划，计算每个终末细胞在各运输阶段接收到的质量分布。

        边际 nu_t[j] 表示终末细胞j在第t个阶段接收到的总质量。
        这些边际描述了每个终末细胞从不同阶段接收到多少"质量流入"，
        帮助我们理解细胞分化过程中质量如何流向终末状态。

        Returns:
            pandas DataFrame:
                - 行：终末细胞的索引
                - 列：nu_1, nu_2, ..., nu_{T-1}（每个阶段的边际）
                - 值：每个终末细胞在每个阶段接收到质量
        """
        # 计算每个运输阶段各终末细胞接收到的质量
        # 对每个阶段k，沿运输矩阵的列方向求和，得到每个终末细胞的边际值
        if self._AUXILIARY_CELL_COST is not None:
            # 存在辅助细胞时，排除最后一列（辅助相关列）
            nu_t = np.array([np.sum(self.transport_plans[k][:,-self._nF:-1],axis=0) for k in range(1,len(self.transport_plans))]).T
        else:
            nu_t = np.array([np.sum(self.transport_plans[k][:,-self._nF:],axis=0) for k in range(1,len(self.transport_plans))]).T

        # 生成列标签：nu_1, nu_2, ..., nu_{T-1}
        nu_labels = ["nu_{t}".format(t=t) for t in range(1,self._T)]

        # 创建DataFrame，行索引为终末细胞
        mass_to_terminals = pd.DataFrame(nu_t, columns=nu_labels, index=self.terminal_cells)
        return mass_to_terminals


    def cost_of_transport(self):
        """计算传输成本（Cost of Transport）

        基于拟合模型的传输计划，计算未正则化问题的总传输成本。

        传输成本是各阶段传输计划与对应代价矩阵的加权求和：
        cost = sum_t sum_ij (P_t[i,j] * C_t[i,j])

        其中P_t是第t阶段的传输计划，C_t是对应的代价矩阵。

        Returns:
            float: 未正则化问题的传输成本
        """
        # 获取各阶段的代价矩阵
        C = self._get_C()

        # 将过大的代价值设为0（避免数值问题）
        # 这些位置通常是辅助细胞相关的条目
        C[0][C[0] > 1e32] = 0
        C[1][C[1] > 1e32] = 0
        C[-1][C[-1] > 1e32] = 0

        # 计算总传输成本：各阶段传输计划与代价矩阵的逐元素乘积求和
        cost = np.sum(self.transport_plans[0]*C[0])  # 第一阶段（初始到中间）
        for t in range(1,self._T-1):
            cost += np.sum(self.transport_plans[t]*C[1])  # 中间各阶段
        cost += np.sum(self.transport_plans[self._T-1]*C[-1])  # 最后阶段（中间到终末）

        return cost


    def max_marginal_groups(self):
        """找到每个细胞的最活跃阶段（Max Marginal Groups）

        基于拟合模型的传输计划，为每个中间细胞找到其发送质量最多的阶段。

        对于每个中间细胞，max_marginal_groups()返回其边际值最大的阶段编号，
        即该细胞在哪个阶段向外输送最多质量。这代表了该细胞在分化过程中的"最活跃"时期。

        如果某细胞的最大边际值很小（< 1e-8），则将其标记为NaN，表示该细胞几乎不参与质量传输。

        Returns:
            dict: 字典 { cell_index : max_group }，其中
                - cell_index: 中间细胞的索引
                - max_group: 该细胞最活跃的阶段编号（从1开始）
                注意：如果细胞的最大边际值小于阈值1e-8，则返回NaN
        """
        # 获取所有细胞的边际分布
        mu = self.marginals()
        cell_annotation = {}

        # 遍历每个中间细胞，找到其最活跃的阶段
        for i in self.intermediate_cells:
            # 如果最大边际值大于阈值1e-8（有效参与质量传输）
            if np.argmax(mu.loc[i]) > 1e-8:
                # 找到最大边际值对应的阶段（argmax返回0-indexed，+1转为1-indexed）
                cell_annotation[i] = np.argmax(mu.loc[i]) + 1
            else:
                # 该细胞几乎不参与质量传输，标记为NaN
                cell_annotation[i] = np.nan
        return cell_annotation



    def transition_matrix(self):
        """构建转移概率矩阵（Transition Matrix）

        基于最优传输计划构建一个转移概率矩阵，用于描述细胞之间的转移关系。

        转移矩阵TM是一个N x N的矩阵（N为细胞总数），其中元素TM[i,j]表示
        从细胞i转移到细胞j的转移概率。

        矩阵结构如下：
        - 前n0行对应初始细胞
        - 中间n行对应中间细胞
        - 最后nF行对应终末细胞（自吸收态，转移到自己的概率为1）

        矩阵构建思路：
        1. 初始细胞：只能转移到中间细胞或终末细胞
        2. 中间细胞：可以相互转移，也可以转移到终末细胞
        3. 终末细胞：只能转移到自身（吸收态）
        4. 每行归一化，使得转移概率之和为1

        Returns:
            numpy ndarray: N x N转移概率矩阵，其中N = n0 + n + nF（细胞总数）
        """
        # 合并中间各阶段的传输计划
        M = np.sum(self.transport_plans[1:-1],axis=0)
        # 添加从中间细胞到终末细胞的传输计划
        M[:,-self._nF:] += self.transport_plans[-1]

        # 构建分块转移矩阵
        if self._AUXILIARY_CELL_COST is not None:
            # 存在辅助细胞时，矩阵包含辅助列
            TM = np.block([
                [np.zeros((self._n0,self._n0)),  self.transport_plans[0][:,:-1], np.zeros((self._n0,self._nF-1)), np.array([self.transport_plans[0][:,-1]]).T],
                [np.zeros((self._n, self._n0)), M],
                [np.zeros((self._nF, self._n0+self._n)), np.eye(self._nF)]
            ] )
        else:
            TM = np.block([
                [np.zeros((self._n0,self._n0)),  self.transport_plans[0], np.zeros((self._n0,self._nF))],
                [np.zeros((self._n, self._n0)), M],
                [np.zeros((self._nF, self._n0+self._n)), np.eye(self._nF)]
            ] )

        # 对每行进行归一化，使得每行之和为1
        TM = np.diag(1/np.sum(TM,axis=1))@TM

        return TM
    
    def mean_absorption_time(self,transition_matrix=None):
        """计算平均吸收时间（Mean Absorption Time）

        基于马尔可夫链模型，计算每个细胞状态的平均吸收时间。

        吸收时间是指从某个初始状态或中间状态出发，到达任意终末状态（吸收态）所需的
        平均转移步数。

        使用线性方程组求解：u = (I - P)^{-1} * 1
        其中P是转移矩阵的瞬态部分（排除终末态），1是全1向量。

        Parameters:
            transition_matrix (numpy ndarray, optional): 预计算的转移矩阵，
                如果未提供则自动计算。默认为None。

        Returns:
            pandas Series: 每个细胞的平均吸收时间
                - 索引：所有细胞的索引（包括初始、中间和终末细胞）
                - 值：平均吸收时间（终末细胞的值为0）
        """
        # 如果未提供转移矩阵，则计算它
        if transition_matrix is None:
            transition_matrix = self.transition_matrix()

        # 提取瞬态部分的转移概率子矩阵（排除终末态）
        I  = np.eye(self._n0+self._n)
        P  = transition_matrix[:self._n0+self._n,:self._n0+self._n]
        _1 = np.ones(P.shape[0])

        # 求解线性方程组 (I - P) * u = 1，得到平均吸收时间
        u  = np.linalg.solve(I-P,_1)
        # 为终末细胞添加0值（已经是吸收态，吸收时间为0）
        u  = np.concatenate((u,np.zeros(self._nF)),axis=0)

        u_df = pd.Series(u)
        # 设置索引标签
        if self._AUXILIARY_CELL_COST is not None:
            u_df.index = self.initial_cells+['auxiliary_initial']+self.intermediate_cells+['auxiliary_intermediate']+self.terminal_cells + ['auxiliary_terminal']
        else:
            u_df.index = self.initial_cells+self.intermediate_cells+self.terminal_cells

        return u_df


    def mean_marginal_group(self):
        """计算每个细胞的平均边际阶段（Mean Marginal Group）

        基于拟合模型的传输计划，为每个细胞计算一个加权平均的边际阶段编号。

        平均边际阶段反映了细胞在分化过程中的"重心位置"——
        一个较早活跃的细胞会有较低的平均阶段，而一个较晚活跃的细胞会有较高的平均阶段。

        计算方法：首先对每个细胞在各阶段的质量进行归一化（使总质量为1），
        然后计算加权平均：mean_group = sum(k * weight_k) for k in 1..T-1

        Returns:
            pandas Series: 每个细胞的平均边际阶段
                - 索引：所有细胞的索引（包括初始、中间和终末细胞）
                - 值：平均边际阶段编号（浮点数）
        """
        # 获取中间细胞在各阶段的边际分布
        mu_df = self.marginals().loc[self.intermediate_cells]

        # 对每个细胞在各阶段的质量进行归一化（使每个细胞的总质量为1）
        mu_df_normed = np.diag(1/mu_df.sum(axis=1))@mu_df

        # 计算加权平均边际阶段
        # 权重向量为 [1, 2, 3, ..., T-1]
        mean_marginals_df = pd.Series(mu_df_normed @ np.array([k for k in range(1,mu_df.shape[1]+1)]), name='mean_marginal_group', index=self.initial_cells+self.intermediate_cells+self.terminal_cells).sort_index()

        return mean_marginals_df



    def pseudotemporal_order(self):
        """计算拟时间排序（Pseudotemporal Order）

        基于平均边际阶段为所有细胞计算拟时间排序，将细胞按其分化成熟度进行排序。

        拟时间是一种介于0到1之间的数值，代表细胞在分化轨迹上的相对位置：
        - 值接近0：较早/较不成熟的细胞（初始细胞）
        - 值接近1：较晚/较成熟的细胞（终末细胞）

        计算方法：
        1. 计算中间细胞的边际质量分布
        2. 对每个细胞的质量在各阶段进行归一化
        3. 计算加权平均阶段作为初步排序依据
        4. 将结果归一化到[0,1]区间

        Returns:
            pandas Series: 每个细胞的拟时间排序
                - 索引：所有细胞的索引（按索引排序）
                - 值：拟时间值，范围[0,1]
                - 初始细胞的拟时间为0
        """
        # 计算终末细胞在各阶段接收的质量分布
        nu_df = self.mass_to_terminals()

        # 对每个终末细胞在各阶段的质量进行归一化（使总质量为1）
        nu_df_normed = np.diag(1/nu_df.sum(axis=1))@nu_df

        # 计算中间细胞在各阶段的质量分布
        mu_df = self.marginals().loc[self.intermediate_cells]

        # 对每个中间细胞在各阶段的质量进行归一化（使总质量为1）
        mu_df_normed = np.diag(1/mu_df.sum(axis=1))@mu_df

        # 计算终末细胞和中间细胞的初步拟时间
        # 权重向量为 [1, 2, 3, ..., T-1]
        pseudotime_mean_terminal = nu_df_normed @ np.array([k for k in range(1,nu_df.shape[1]+1)])
        pseudotime_mean_intermediate = mu_df_normed @ np.array([k for k in range(1,mu_df.shape[1]+1)])

        # 合并中间细胞和终末细胞的初步拟时间
        pseudotime_mean = pd.concat((pseudotime_mean_intermediate,pseudotime_mean_terminal)).reset_index(drop=True)

        # 按初步拟时间排序并计算排名
        temporal_ordering_normed_mid_end = pd.DataFrame(pseudotime_mean.sort_values())
        temporal_ordering_normed_mid_end['order'] = [k for k in range(1, temporal_ordering_normed_mid_end.shape[0]+1)]
        # 将排名归一化到[0,1]区间
        temporal_ordering_normed_mid_end = temporal_ordering_normed_mid_end.sort_index()['order']/(temporal_ordering_normed_mid_end.shape[0])

        # 为初始细胞添加0值（最不成熟的细胞）
        temporal_ordering_normed = np.concatenate((np.zeros(len(self.initial_cells)),temporal_ordering_normed_mid_end.values))

        # 创建最终的拟时间Series
        temporal_ordering_normed_df = pd.Series(temporal_ordering_normed, name='pseudotime',index=self.initial_cells+self.intermediate_cells+self.terminal_cells).sort_index()

        return temporal_ordering_normed_df
    
    def _get_global_to_local_index_dict(self):

        global_indices = self.initial_cells + self.intermediate_cells + self.terminal_cells

        return {global_indices[k] : k for k in range(len(self.initial_cells + self.intermediate_cells + self.terminal_cells))}

    def cell_fate_probabilities(self, fate_groups = None, transition_matrix=None):
        """计算细胞命运概率（Cell Fate Probabilities）

        基于最优传输计划构建的马尔可夫链模型，计算每个细胞最终被吸收到各个终末命运类别的概率。

        吸收概率A[i,j]表示从细胞i出发，最终被吸收到终末状态j的概率。
        这些概率揭示了每个细胞对各种终末命运的"倾向性"。

        使用线性方程组求解：A = (I - P)^{-1} * S
        其中P是瞬态部分的转移子矩阵，S是瞬态到吸收态的转移子矩阵。

        Parameters:
            fate_groups (dict): 命运分组字典，格式为 { fate_label : index_array }
                - fate_label (str): 命运名称（如'erythroid'）
                - index_array: 对应的终末细胞索引数组
            transition_matrix (numpy ndarray, optional): 预计算的转移矩阵，
                如果未提供则自动计算。默认为None。

        Returns:
            pandas DataFrame: 细胞命运概率矩阵
                - 行：所有细胞的索引
                - 列：每个命运标签
                - 值：每个细胞被吸收到各个命运的概率
                注意：如果启用了辅助细胞，还会包含"Fate unknown"列
        """
        # 构建单位矩阵
        I  = np.eye(self._n0+self._n)
        # 如果未提供转移矩阵，则计算它
        if transition_matrix is None:
            transition_matrix = self.transition_matrix()

        # 提取瞬态部分子矩阵P和瞬态到吸收态的转移子矩阵S
        P  = transition_matrix[:self._n0+self._n,:self._n0+self._n]
        S  = transition_matrix[:self._n0+self._n,self._n0+self._n:]

        # 求解线性方程组 (I - P) * A = S，得到吸收概率矩阵A
        # A[i,j]表示从细胞i出发被吸收到终末状态j的概率
        A  = np.linalg.solve(I-P,S) #Solves a linear system of equations for the absoprtion probabilities A_{ij}.
                                    #Where i is a cell index, and j a terminal cell state.

        # 为终末细胞添加吸收概率（终末细胞自身被吸收的概率为1）
        A  = np.concatenate((A, np.eye(self._nF)),axis=0)
        A_df = pd.DataFrame(A)

        # 设置列标签为终末细胞索引
        if self._AUXILIARY_CELL_COST is not None:
            A_df.columns = self.terminal_cells + ['auxiliary_terminal']
        else:
            A_df.columns = self.terminal_cells

        # 按命运分组汇总吸收概率
        absorption_probabilities = pd.DataFrame()
        for fate_label in fate_groups:
            # 对于每个命运标签，汇总该组内所有终末细胞的吸收概率
            absorption_probabilities[fate_label] = A_df.loc[:,fate_groups[fate_label]].sum(axis=1)

        # 如果启用了辅助细胞，添加"Fate unknown"列（表示无法明确分类的细胞）
        if self._AUXILIARY_CELL_COST is not None:
            absorption_probabilities['Fate unknown'] = A_df.loc[:,'auxiliary_terminal'].values

        # 设置行索引
        if self._AUXILIARY_CELL_COST is not None:
            absorption_probabilities.index = self.initial_cells+['auxiliary_initial']+self.intermediate_cells+['auxiliary_intermediate']+self.terminal_cells + ['auxiliary_terminal']
        else:
            absorption_probabilities.index = self.initial_cells+self.intermediate_cells+self.terminal_cells
            
        return absorption_probabilities




    def proximal_sinkhorn(self, epsilon_threshold : float = None, patience : int = 100, verbose : bool = False):
        """执行近端Sinkhorn方案逐步降低正则化参数

        通过迭代地使用近端Sinkhorn算法，逐步减小有效正则化参数epsilon，
        从而逼近无正则化的最优传输解。

        算法背景:
        熵正则化OT问题的解与正则化参数epsilon密切相关：
        - 较大的epsilon：解更平滑，但偏离真实OT解
        - 较小的epsilon：解更接近真实OT，但数值上更难求解

        直接求解小epsilon的问题会导致数值不稳定。近端Sinkhorn方案通过以下策略解决：
        1. 从较大的epsilon开始，使用标准Sinkhorn求解
        2. 将当前解作为"先验"(prior)，使用proximal Sinkhorn求解下一个较小的epsilon
        3. 重复直到epsilon达到目标阈值

        有效epsilon的更新规则:
        1/epsilon_new = 1/epsilon_old + 1/epsilon_proximal
        这确保了每次迭代后，有效epsilon逐渐减小

        收敛性:
        当 total_epsilon >= epsilon_threshold 时算法终止
        其中 total_epsilon = 1/(1/epsilon_old + 1/epsilon_proximal)

        Parameters:
            -----------
            epsilon_threshold : float
                目标epsilon值，算法将在达到此阈值时停止。
                通常设置为一个较小的值（如0.001），以获得接近无正则化的解。
            patience : int
                Sinkhorn迭代中监控进度的频率（默认100）。
                每patience次迭代打印一次进度信息。
            verbose : bool
                是否输出详细的迭代信息（默认False）。

        Returns:
            None（结果存储在以下实例属性中）:
            - self.transport_plans: 最终的传输计划
            - self.dual_variables: 最终的对偶变量
            - self._EPSILON: 最终的有效正则化参数
            - self.history: 包含proximal Sinkhorn迭代历史的字典
        """
        # =====================================================================
        # 初始化：保存初始的工具变量作为第一个先验
        # =====================================================================
        self.__prior_utility_variables = [copy.deepcopy(self.utility_variables)]


        np.seterr(all='ignore')


        start_time = time.time()
        iterations = 0

        # =====================================================================
        # 初始化proximal epsilon历史记录
        # =====================================================================
        if self._PROXIMAL_EPSILON is None:
            self._PROXIMAL_EPSILON = self._EPSILON

            self._PROXIMAL_EPSILON_HISTORY.append(self._PROXIMAL_EPSILON)
            self._PROXIMAL_EPSILON_HISTORY.append(self._PROXIMAL_EPSILON) # 第一个分量对应迭代0，第二个对应一次"标准"Sinkhorn解后
            self._TOTAL_EPSILON_HISTORY.append(self._EPSILON)
            self._TOTAL_EPSILON_HISTORY.append(self._EPSILON)             # 第一个分量对应迭代0，第二个对应一次"标准"Sinkhorn解后

            self._INNER_ITERATIONS = [0,patience*(len(self.history['infeasibility']))]
            self._OUTER_PROXIMAL_ITERATIONS = 0


        # 记录初始epsilon（用于计算收敛进度）
        first_epsilon = self._EPSILON

        # =====================================================================
        # 主循环：迭代调用fit方法，每次使用前一次的传输计划作为先验
        # =====================================================================
        r = 0
        while r < 1:

            # 调用fit方法，使用当前传输计划作为先验
            # fit方法内部会调用_sinkhorn_iterations_with_prior
            self.fit(self.data,
                                    prior     = self.transport_plans,
                                    verbose   = verbose,
                                    patience  = patience,
                                    tolerance = self._TOLERANCE,
                            )

            # 保存当前的工具变量（作为下一次迭代的先验）
            self.__prior_utility_variables.append(copy.deepcopy(self.utility_variables))


            elapsed_time = time.time() - start_time

            # 记录当前的总epsilon
            total_epsilon = self._EPSILON
            self._TOTAL_EPSILON_HISTORY.append(total_epsilon)

            self._INNER_ITERATIONS.append(patience*(len(self.history['infeasibility'])))

            # 记录当前的proximal epsilon
            current_epsilon = self._PROXIMAL_EPSILON
            self._PROXIMAL_EPSILON_HISTORY.append(current_epsilon)

            # 计算收敛进度 r = (当前epsilon - 初始epsilon) / (目标epsilon - 初始epsilon)
            r = (total_epsilon - first_epsilon)/(epsilon_threshold - first_epsilon)


            self._OUTER_PROXIMAL_ITERATIONS += 1
            iterations += 1

            # 打印进度信息
            print("\r", "[Proximal Sinkhorn] Outer iterations: {iter}".format(iter=iterations) + " | Initial epsilon: {0:.4e}".format(first_epsilon)+ " | Current epsilon: {0:.4e}".format(current_epsilon) + " | Total epsilon: {0:.4e}".format(total_epsilon) + " | Elapsed time: {time}".format(time=timedelta(seconds=elapsed_time)), end = "", flush=True)

        # =====================================================================
        # 保存proximal Sinkhorn的历史记录
        # =====================================================================
        self.history['proximal_sinkhorn'] = {'iterations' : self._INNER_ITERATIONS, 'proximal_epsilon' : self._PROXIMAL_EPSILON_HISTORY, 'total_epsilon' : self._TOTAL_EPSILON_HISTORY}

        print("\n Terminating proximal scheme.", flush=True)


        return


