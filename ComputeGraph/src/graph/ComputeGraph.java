package graph;

import java.util.Hashtable;
import java.util.List;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;

import function.Function;
import matrix.Matrix;
import vertex.ComputeNode;

public class ComputeGraph extends SingleGraph
{
	private List<String> inputVertices;
	private Hashtable<String, String> outputVertices;
	
	public ComputeGraph(String name)
	{
		super(name);
	}
	
	public void setInputVertices(List<String> newInputVertices)
	{
		inputVertices=newInputVertices;
	}
	
	public void setOutputVertices(List<String> newOutputVertices)
	{
		outputVertices=new Hashtable<>();
		for(String outputName: newOutputVertices)
		{
			outputVertices.put(outputName, outputName);
		}
	}
	
	public void addNode(String name, Function function)
	{
		ComputeNode newNode=addNode(name);
		newNode.setFunction(function);
	}
	
	public Hashtable<String, Matrix> getOutput(Hashtable<String, Matrix> inputs)
	{
		Hashtable<String, String> outputted=new Hashtable<>();
		Hashtable<String, Matrix> intermediateOutputs=new Hashtable<>();
		Hashtable<String, Matrix> outputs=new Hashtable<>();
		for(String inputName: inputs.keySet())
		{
			if(outputVertices.get(inputName)==null)
			{
				intermediateOutputs.put(inputName, inputs.get(inputName));
			}
			else
			{
				outputs.put(inputName, inputs.get(inputName));
			}
			outputted.put(inputName, inputName);
		}
		
		while(intermediateOutputs.size()>0)
		{
			for(String intermediateNode: intermediateOutputs.keySet())
			{
				for(Edge computedNodeEdge: getNode(intermediateNode).getEachEdge())
				{
					if(outputted.get(computedNodeEdge.getNode1().getId())==null
							&& !computedNodeEdge.getNode1().getId().equals(intermediateNode))
					{
						ComputeNode nextNode=getNode(computedNodeEdge.getNode1().getId());
						boolean allComputed=true;
						Hashtable<String, Matrix> nextNodeInput=new Hashtable<>();
						for(Edge nextNodeEdge: nextNode.getEachEdge())
						{
							if(outputted.get(computedNodeEdge.getNode0().getId())==null)
							{
								allComputed=false;
							}
							else
							{
								nextNodeInput.put(computedNodeEdge.getNode0().getId(),
										intermediateOutputs.get(computedNodeEdge.getNode0().getId()));
							}
						}
						if(allComputed)
						{
							Matrix nextNodeOutput=nextNode.getOutput(nextNodeInput);
							intermediateOutputs.put(nextNode.getId(), nextNodeOutput);
							if(outputVertices.get(nextNode.getId())!=null)
							{
								outputs.put(nextNode.getId(), nextNodeOutput);
							}
						}
					}
				}
			}
		}
		return outputs;
	}

}
