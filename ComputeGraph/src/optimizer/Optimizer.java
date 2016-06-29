package optimizer;

import graph.ComputeGraph;

public abstract class Optimizer 
{
	
	public abstract void optimize(ComputeGraph cg, String objective);

}
