from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np
from heapq import heappush,heappop,nsmallest

class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):
        """
        TODO: implement this method!
        """

        if self.close.has_state(successor_node.state) or self.open.has_state(successor_node.state):
            return

        self.open.push_node(successor_node)

    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        TODO: implement this method!
        Remember: `GreedyStochastic` is greedy.
        """
        h = self.heuristic_function.estimate
        return h(search_node.state)

    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        TODO: implement this method!
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """
        if self.open.is_empty():
            return None

        min_size = min(self.N, len(self.open))
        best_heuristics = []
        nodes = []

        #iterating over the min_size minimal values of heuristics,
        # and creating an array of tuples (h(state),node)
        for i in range(0,min_size):
            node = self.open.pop_next_node()
            nodes.append(node)
            best_heuristics.append((node.expanding_priority,node))

        for i in range(0,len(nodes)):
            self.open.push_node(nodes[i])

        min_tuple = min(best_heuristics,key=lambda x:x[0]) #get tuple with minimal heuristic
        min_h = min_tuple[0] #minimal heuristic value
        node_to_expand = min_tuple[1]

        if (min_h != 0): #if not target node, calculate probabilities
            sum1 = sum(map(lambda tuple: ((tuple[0] / min_h) ** (-1 / self.T)), best_heuristics))
            probabilities = list(map(lambda tuple: ((tuple[0] / min_h) ** (-1 / self.T) / sum1), best_heuristics))
            chosen_tuple_idx = np.random.choice(min_size, p=probabilities)
            node_to_expand = best_heuristics[chosen_tuple_idx][1]

        self.open.extract_node(node_to_expand)
        if self.use_close:
            self.close.add_node(node_to_expand)

        self.T *=self.T_scale_factor
        return node_to_expand


