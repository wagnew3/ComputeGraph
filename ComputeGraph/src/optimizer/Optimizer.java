package optimizer;

import graph.ComputeGraph;
import vertex.ComputeNode;

public abstract class Optimizer 
{
	
	public abstract void optimize(ComputeGraph cg, ComputeNode[] objective);

}
