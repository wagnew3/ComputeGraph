package graph;

import java.util.Hashtable;
import java.util.List;

import org.graphstream.graph.Edge;
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
	
	@Override
	public Edge addEdge(String id, String index1, String index2)
	{
		Edge edge=super.addEdge(id+" "+index1+"_"+index2, index1, index2);
		return edge;
	}
	
	public Edge addEdge(String index1, String index2)
	{
		Edge edge=super.addEdge("in "+index1+"_"+index2, index1, index2);
		return edge;
	}
	
	public Hashtable<String, Matrix> getOutput(Hashtable<String, Matrix> inputs)
	{
		Hashtable<String, Matrix> outputs=new Hashtable<>();
		Hashtable<String, Matrix> allOutputs=compute(inputs);
		for(String outputName: outputVertices.keySet())
		{
			outputs.put(outputName, allOutputs.get(outputName));
		}
		return outputs;
	}
	
	public Hashtable<String, Matrix> compute(Hashtable<String, Matrix> inputs)
	{
		Hashtable<String, String> outputted=new Hashtable<>();
		Hashtable<String, Matrix> intermediateOutputs=new Hashtable<>();
		for(String inputName: inputs.keySet())
		{
			if(outputVertices.get(inputName)==null)
			{
				intermediateOutputs.put(inputName, inputs.get(inputName));
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
							if(outputted.get(nextNodeEdge.getNode0().getId())==null)
							{
								allComputed=false;
							}
							else
							{
								String name=nextNodeEdge.getId().substring(0, nextNodeEdge.getId().lastIndexOf(' '));
								nextNodeInput.put(name,
										intermediateOutputs.get(nextNodeEdge.getNode0().getId()));
							}
						}
						if(allComputed)
						{
							Matrix nextNodeOutput=nextNode.getOutput(nextNodeInput);
							intermediateOutputs.put(nextNode.getId(), nextNodeOutput);
						}
					}
				}
			}
		}
		return intermediateOutputs;
	}

}
