from framework.graph_search import *
from .map_problem import MapProblem, MapState


class AirDistHeuristic(HeuristicFunction):
    heuristic_name = 'AirDist'

    def estimate(self, state: GraphProblemState) -> float:
        """
        The air distance between the geographic location represented
         by `state` and the geographic location of the problem's target.

        TODO: implement this method!
        Use `self.problem` to access the problem.
        Use `self.problem.roads` to access the map.
        Given a junction index, use `roads[junction_id]` to find the
         junction instance (of type Junction).
        Use the method `calc_air_distance_from()` to calculate the
        air distance between two junctions.
        """
        #Ex. 11
        assert isinstance(self.problem, MapProblem)
        assert isinstance(state, MapState)

        target_junction = self.problem.roads[self.problem.target_junction_id]
        current_junction = self.problem.roads[state.junction_id]
        return target_junction.calc_air_distance_from(current_junction)

        #raise NotImplemented()

