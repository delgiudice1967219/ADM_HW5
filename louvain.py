from .community import Community


class Louvain:
    def __init__(self, graph, weight='weight', show=False) -> None:
        self.show = show
        self.graph = graph
        # Convert directed graph to undirected if necessary
        if self.graph.is_directed():
            print("Converting directed graph to undirected.")
            self.graph = self.graph.to_undirected()
        self.weight = weight
        self.community_map = {}
        self.iterations = 0
        self.done = False
        self.community = Community(graph, weight)
    
    def iteration(self):
        """Perform a single iteration of the Louvain method."""
        self.iterations += 1
        if self.show:
            print(f"Iteration {self.iterations}")
        
        self.community.initialize()  # Initialize the community tracking system
        no_change = True
        improved = True
        cycle_iter = 0
        total_nodes_changed = 0

        while improved:
            improved = False  # Reset the improved flag for this iteration
            nodes_changed_in_this_cycle = 0  # Track changes for the current cycle
            cycle_iter += 1
            
            for node in self.community:  # This uses the __iter__ method of the Community class
                modified = self.community.assign_best_community(node)
                if modified:
                    improved = True
                    no_change = False
                    nodes_changed_in_this_cycle += 1  # Increment changes in the current cycle
                
            total_nodes_changed += nodes_changed_in_this_cycle  # Update the total changes for this iteration
        
        # Print total number of nodes changed in the current iteration after all cycles
        if self.show:
            print(f"Iteration {self.iterations} - Total nodes changed: {total_nodes_changed}")
        
        self.community.update_result_map()
        self.community.agglomerate()

        if no_change:
            self.done = True

    def run(self):
        """Run the Louvain algorithm until convergence."""
        while not self.done:
            self.iteration()
        self.community_map = self.community.result_map
        return self.community_map
