package graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.NodeFactory;
import org.graphstream.graph.implementations.AbstractGraph;
import org.graphstream.graph.implementations.SingleGraph;
import org.graphstream.graph.implementations.SingleNode;

import function.DifferentialbleFunction;
import function.Function;
import matrix.Matrix;
import vertex.ComputeNode;

public class ComputeGraph extends SingleGraph
{
	private Hashtable<String, String> outputVertices;
	List<String> computeOrder;
	
	public ComputeGraph(String name)
	{
		super(name);
		computeOrder=null;
		setNodeFactory(new NodeFactory<SingleNode>() 
		{
			public SingleNode newInstance(String id, Graph graph) 
			{
				return new ComputeNode((AbstractGraph) graph, id);
			}
		});
	}
	
	public void setOutputVertices(List<String> newOutputVertices)
	{
		computeOrder=null;
		outputVertices=new Hashtable<>();
		for(String outputName: newOutputVertices)
		{
			outputVertices.put(outputName, outputName);
		}
	}
	
	public void addNode(String name, Function function)
	{
		computeOrder=null;
		ComputeNode newNode=addNode(name);
		newNode.setFunction(function);
	}
	
	@Override
	public Edge addEdge(String id, String index1, String index2)
	{
		computeOrder=null;
		Edge edge=super.addEdge(id+" "+index1+"_"+index2, index1, index2);
		return edge;
	}
	
	public Edge addEdge(String index1, String index2)
	{
		computeOrder=null;
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
		if(computeOrder==null)
		{
			computeOrder=determineComputeOrder();
		}
		
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
		
		for(String toCompute: computeOrder)
		{
			if(outputted.get(toCompute)==null)
			{
				Hashtable<String, Matrix> nextNodeInput=new Hashtable<>();
				ComputeNode nextComputeNode=getNode(toCompute);
				for(Edge computedNodeEdge: nextComputeNode)
				{
					if(computedNodeEdge.getNode1().getId().equals(toCompute))
					{
						nextNodeInput.put(computedNodeEdge.getId().substring(0, computedNodeEdge.getId().indexOf(' ')),
								intermediateOutputs.get(computedNodeEdge.getNode0().getId()));
					}
				}
				Matrix nextNodeOutput=nextComputeNode.getOutput(nextNodeInput);
				intermediateOutputs.put(toCompute, nextNodeOutput);
				outputted.put(toCompute, toCompute);
			}
		}
		return intermediateOutputs;
	}
	
	public Hashtable<String, Matrix>[] derive(Hashtable<String, Matrix> inputs)
	{
		Hashtable<String, Matrix> allOutputs=compute(inputs);
		
		Hashtable<String, Matrix> objectiveDerivatives=new Hashtable<>();
		Hashtable<String, Matrix> parameterDerivatives=new Hashtable<>();
		
		for(int computeInd=computeOrder.size()-1; computeInd>=0; computeInd--)
		{
			String toCompute=computeOrder.get(computeInd);
			Hashtable<String, Matrix> nextNodeInputs=new Hashtable<>();
			Hashtable<String, Matrix> derivatives=new Hashtable<>();
			ComputeNode nextComputeNode=getNode(toCompute);
			if(nextComputeNode.getFunction() instanceof DifferentialbleFunction)
			{
				for(Edge computedNodeEdge: nextComputeNode)
				{
					if(computedNodeEdge.getNode1().getId().equals(toCompute))
					{
						nextNodeInputs.put(computedNodeEdge.getId().substring(0, computedNodeEdge.getId().indexOf(' ')),
								allOutputs.get(computedNodeEdge.getNode0().getId()));
					}
					else
					{
						if(objectiveDerivatives.get(computedNodeEdge.getNode1().getId())!=null)
						{
							derivatives.put(computedNodeEdge.getId().substring(0, computedNodeEdge.getId().indexOf(' ')),
									objectiveDerivatives.get(computedNodeEdge.getNode1().getId()));
						}
					}
				}
				Matrix[] nextNodeDerivatives=((DifferentialbleFunction)nextComputeNode.getFunction())
						.differentiate(nextNodeInputs, derivatives);
				objectiveDerivatives.put(nextComputeNode.getId(), nextNodeDerivatives[0]);
				if(nextNodeDerivatives[1]!=null)
				{
					parameterDerivatives.put(nextComputeNode.getId(), nextNodeDerivatives[1]);
				}
			}
		}
		return new Hashtable[]{objectiveDerivatives, parameterDerivatives};
	}
	
	private List<String> determineComputeOrder()
	{
		Hashtable<String, String> addedNode=new Hashtable<>();
		List<String> newComputeOrder=new ArrayList<>();
		List<String> nodesToComputeFrom=new ArrayList<>();
		
		for(String node: outputVertices.keySet())
		{
			ComputeNode outputNode=getNode(node);
			boolean terminal=true;
			for(Edge edge: outputNode.getEdgeSet())
			{
				if(!edge.getNode1().getId().equals(node)
						&& outputVertices.get(edge.getNode1().getId())!=null)
				{
					terminal=false;
				}
			}
			newComputeOrder.add(node);
			nodesToComputeFrom.add(node);
			addedNode.put(node, node);
		}
		
		while(!nodesToComputeFrom.isEmpty())
		{
			String computeFromNodeID=nodesToComputeFrom.remove(0);
			ComputeNode computeNode=getNode(computeFromNodeID);
			boolean terminal=true;
			for(Edge edge: computeNode.getEdgeSet())
			{
				if(edge.getNode1().getId().equals(computeNode.getId())
						&& addedNode.get(edge.getNode0().getId())==null)
				{
					newComputeOrder.add(edge.getNode0().getId());
					nodesToComputeFrom.add(edge.getNode0().getId());
					addedNode.put(edge.getNode0().getId(), edge.getNode0().getId());
				}
			}
		}
		
		Collections.reverse(newComputeOrder);
		return newComputeOrder;
	}

}
