from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem
from .map_heuristics import AirDistHeuristic

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)
        old_state = state_to_expand

        source_junction = old_state.current_location
        possible_stop_stations = self.possible_stop_points - old_state.dropped_so_far

        for target_junction in possible_stop_stations:
            #calculate cost between two junctions
            fuel_cost = self._get_from_cache((source_junction.index,target_junction.index))
            if fuel_cost is None:  # didn't find junction in cache
                map_prob = MapProblem(self.roads, source_junction.index, target_junction.index)
                astar = self.inner_problem_solver
                astar_res = astar.solve_problem(map_prob)
                fuel_cost = astar_res.final_search_node.cost
                self._insert_to_cache((source_junction.index,target_junction.index), fuel_cost)

            if old_state.fuel < fuel_cost: #not enough fuel, continue search
                continue

            if target_junction in self.gas_stations:
                new_dropped_so_far = old_state.dropped_so_far
                new_fuel = self.gas_tank_capacity
            elif target_junction in self.drop_points:
                new_dropped_so_far = old_state.dropped_so_far | {target_junction}
                new_fuel = old_state.fuel - fuel_cost
                if new_fuel == 0: #will be stuck in the target junction, continue search
                    continue

            new_state = StrictDeliveriesState(target_junction, new_dropped_so_far, new_fuel)
            yield (new_state, fuel_cost)


    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        TODO: implement this method!
        """
        assert isinstance(state, StrictDeliveriesState)

        left_to_drop = self.drop_points.difference(state.dropped_so_far)
        return len(left_to_drop) == 0 and state.current_location in self.drop_points
