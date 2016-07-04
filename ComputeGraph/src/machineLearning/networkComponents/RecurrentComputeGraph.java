package machineLearning.networkComponents;

import java.util.ArrayList;
import java.util.List;

import graph.ComputeGraph;
import vertex.ComputeNode;

public abstract class RecurrentComputeGraph extends ComputeGraph
{
	
	public List<ComputeNode> initialState;
	public List<ComputeNode> inputNodes;
	public List<ComputeNode> memoryInNodes;
	public List<ComputeNode> memoryOutNodes; 
	public List<ComputeNode> computeOuts;
	public List<ComputeNode> objectives; 
	public List<ComputeNode> trainingOutputNodes;

	public RecurrentComputeGraph(String name)
	{
		super(name);
		initialState=new ArrayList<>();
		inputNodes=new ArrayList<>();
		memoryInNodes=new ArrayList<>();
		memoryOutNodes=new ArrayList<>();
		computeOuts=new ArrayList<>();
		objectives=new ArrayList<>();
		trainingOutputNodes=new ArrayList<>();
	}

}
