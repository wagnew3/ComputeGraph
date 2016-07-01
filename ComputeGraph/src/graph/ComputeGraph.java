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

import function.DifferentiableFunction;
import function.Function;
import matrix.Matrix;
import vertex.ComputeNode;

public class ComputeGraph extends SingleGraph
{
	private Hashtable<ComputeNode, ComputeNode> outputVertices;
	List<ComputeNode> computeOrder;
	
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
	
	public void setOutputVertices(List<ComputeNode> newOutputVertices)
	{
		computeOrder=null;
		outputVertices=new Hashtable<>();
		for(ComputeNode outputName: newOutputVertices)
		{
			outputVertices.put(outputName, outputName);
		}
	}
	
	public ComputeNode addNode(String name, Function function)
	{
		computeOrder=null;
		ComputeNode newNode=addNode(name);
		newNode.setFunction(function);
		return newNode;
	}
	
	public Edge addEdge(ComputeNode c0, ComputeNode c1)
	{
		if(!c0.hasEdgeBetween(c1))
		{
			computeOrder=null;
			Edge edge=super.addEdge(c0.toString()+"_"+c1.toString(), c0, c1);
			return edge;
		}
		return null;
	}
	
	public Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> getOutput(Hashtable<ComputeNode, Matrix> inputs)
	{
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> outputs=new Hashtable<>();
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> allOutputs=compute(inputs);
		for(ComputeNode outputName: outputVertices.keySet())
		{
			outputs.put(outputName, allOutputs.get(outputName));
		}
		return outputs;
	}
	
	public Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> compute(Hashtable<ComputeNode, Matrix> inputs)
	{
		if(computeOrder==null)
		{
			computeOrder=determineComputeOrder();
		}
		
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> allInputs=new Hashtable<>();
		for(ComputeNode inputNode: inputs.keySet())
		{
			Hashtable inputNodeInput=new Hashtable<>();
			inputNodeInput.put(inputNode, inputs.get(inputNode));
			allInputs.put(inputNode, inputNodeInput);
			/*
			Hashtable<ComputeNode, Matrix> input=new Hashtable<>();
			input.put(inputNode, inputs.get(inputNode));
			Hashtable<ComputeNode, Matrix> nodeOutputs=inputNode.getOutput(input);
			for(ComputeNode nodeOutputTo: nodeOutputs.keySet())
			{
				if(allInputs.get(nodeOutputTo)==null)
				{
					allInputs.put(nodeOutputTo, new Hashtable<ComputeNode, Matrix>());
				}
				allInputs.get(nodeOutputTo).put(inputNode, nodeOutputs.get(nodeOutputTo));
			}
			*/
		}
		
		for(ComputeNode toCompute: computeOrder)
		{
			Hashtable<ComputeNode, Matrix> nodeOutputs=toCompute.getOutput(allInputs.get(toCompute));
			for(ComputeNode nodeOutputTo: nodeOutputs.keySet())
			{
				if(allInputs.get(nodeOutputTo)==null)
				{
					allInputs.put(nodeOutputTo, new Hashtable<ComputeNode, Matrix>());
				}
				allInputs.get(nodeOutputTo).put(toCompute, nodeOutputs.get(nodeOutputTo));
			}
		}
		return allInputs;
	}
	
	public Hashtable<ComputeNode, Matrix> derive(Hashtable<ComputeNode, Matrix> inputs)
	{
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> allInputs=compute(inputs);
		
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> objectiveDerivatives=new Hashtable<>();
		Hashtable<ComputeNode, Matrix> parameterDerivatives=new Hashtable<>();
		
		for(int computeInd=computeOrder.size()-1; computeInd>=0; computeInd--)
		{
			ComputeNode nextComputeNode=computeOrder.get(computeInd);
			Hashtable<String, Matrix> derivatives=new Hashtable<>();
			if(nextComputeNode.getFunction() instanceof DifferentiableFunction)
			{
				Hashtable<ComputeNode, Matrix>[] nextNodeDerivatives
					=nextComputeNode.differentiate(allInputs.get(nextComputeNode), objectiveDerivatives.get(nextComputeNode));
				
				for(ComputeNode prevNode: nextNodeDerivatives[0].keySet())
				{
					if(objectiveDerivatives.get(prevNode)==null)
					{
						objectiveDerivatives.put(prevNode, new Hashtable<ComputeNode, Matrix>());
					}
					objectiveDerivatives.get(prevNode).put(nextComputeNode, nextNodeDerivatives[0].get(prevNode));
				}
				if(nextNodeDerivatives[1]!=null)
				{
					for(ComputeNode paramComputeNode: nextNodeDerivatives[1].keySet())
					{
						parameterDerivatives.put(paramComputeNode, nextNodeDerivatives[1].get(paramComputeNode));
					}
				}
			}
		}
		return parameterDerivatives;
	}
	
	private List<ComputeNode> determineComputeOrder()
	{
		Hashtable<ComputeNode, ComputeNode> addedNode=new Hashtable<>();
		List<ComputeNode> newComputeOrder=new ArrayList<>();
		List<ComputeNode> nodesToComputeFrom=new ArrayList<>();
		
		for(ComputeNode outputNode: outputVertices.keySet())
		{
			boolean terminal=true;
			for(Edge edge: outputNode.getEdgeSet())
			{
				if(!edge.getNode1().getId().equals(outputNode)
						&& outputVertices.get(edge.getNode1().getId())!=null)
				{
					terminal=false;
				}
			}
			newComputeOrder.add(outputNode);
			nodesToComputeFrom.add(outputNode);
			addedNode.put(outputNode, outputNode);
		}
		
		while(!nodesToComputeFrom.isEmpty())
		{
			ComputeNode computeNode=nodesToComputeFrom.remove(0);
			boolean terminal=true;
			for(Edge edge: computeNode.getEdgeSet())
			{
				if(edge.getNode1().equals(computeNode)
						&& addedNode.get(edge.getNode0())==null)
				{
					newComputeOrder.add(edge.getNode0());
					nodesToComputeFrom.add(edge.getNode0());
					addedNode.put(edge.getNode0(), edge.getNode0());
				}
			}
		}
		
		Collections.reverse(newComputeOrder);
		return newComputeOrder;
	}

}
