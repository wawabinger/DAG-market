# Supplementary Material

## 1. Theoretical proofs of greedt property

To solve the above problem, we design a greedy random walk algorithm (GRWA)  to greedily select models with lower prices and higher similarity. But first, we prove that this problem can be solved using a greedy algorithm if it satisfy the following property.

**1.1 Greedy Choice Property**

**Theorem**

The optimal solution to the problem can be achieved by making a locally optimal choice at each step.

**Proof**

$\tfrac{P(v_{i+1})}{C(v_{i+1})}$ can be represented by $P(v_{i+1}) \cdot C^{-1}(v_{i+1})$. Let the current path be $\varPhi = (v_1, v_2, \dots, v_i)$. When selecting the next node $v_{i+1}$, we choose the node that maximizes $P(v_{i+1}) \cdot C^{-1}(v_{i+1})$. Suppose there exists another path $\varPhi' = (v_1', v_2', \dots, v_i', v_{i+1}')$ such that $\sum_{v_j' \in \varPhi'} P(v_{j}') \cdot C^{-1}(v_{j}') > \sum_{v_j \in \varPhi \cup \{v_{i+1}\}} P(v_{j}) \cdot C^{-1}(v_{j})$.

If $v_{i+1}' \neq v_{i+1}$, according to the definition of greedy choice, we have $P(v_{i+1}) \cdot C^{-1}(v_{i+1}) \le P(v_{i+1}') \cdot C^{-1}(v_{i+1}')$. Therefore,

​    $\sum_{v_j \in \varPhi} P(v_{j}) \cdot C^{-1}(v_{j}) + P(v_{i+1}) \cdot C^{-1}(v_{i+1}) \\
​    \geq \sum_{v_j \in \varPhi} P(v_{j}) \cdot C^{-1}(v_{j}) + P(v_{i+1}') \cdot C^{-1}$(v_{i+1}')

Thus,

​    $\sum_{v_j \in \varPhi} P(v_{j}) \cdot C^{-1}(v_{j}) > \sum_{v_j' \in \varPhi'} P(v_{j}') \cdot C^{-1}(v_{j}')$

This contradicts the assumption. Therefore, selecting the node that minimizes $P(v_{i+1}) \cdot C^{-1}(v_{i+1})$ is the correct locally optimal choice.
\end{proof}

**1.2 Optimal Substructure Property**

**Theorem**

An optimal solution to the problem contains optimal solutions to its subproblems.
**proof**
    Let $\varPhi^* = (v_1, v_2, \dots, v_k)$ be the globally optimal path. Suppose its subpath $\varPhi' = (v_1', v_2', \dots, v_i')$ is also the optimal path for the corresponding subproblem.

For any node $v_{i+1}$, after choosing the node that minimizes $P(v_{i+1}) \cdot C^{-1}(v_{i+1})$, we have:

​    $\sum_{v_j \in \varPhi'} P(v_j) \cdot C^{-1}(v_j) + P(v_{i+1}) \cdot C^{-1}(v_{i+1})\\
​    = \sum_{v_j \in \varPhi^*} P(v_j) \cdot C^{-1}(v_j)$

Thus,

​    $\sum_{v_j' \in \varPhi'} P(v_j') \cdot C^{-1}(v_j') +  P(v_{i+1}) \cdot C^{-1}(v_{i+1})\\
​    = \sum_{v_j \in \varPhi^*} P(v_j) \cdot C^{-1}(v_j)$

Therefore, $\varPhi^* = \varPhi' \cup \{v_{i+1}\}$ is part of the globally optimal path. Thus, the optimal solution to the problem contains the optimal solutions to its subproblems.
