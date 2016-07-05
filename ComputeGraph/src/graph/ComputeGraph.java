package graph;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import org.graphstream.graph.Edge;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
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
	
	public void removeNode(ComputeNode node)
	{
		super.removeNode(node);
		ComputeNode toRemove=(ComputeNode)node;
		for(Node otherNode: getNodeSet())
		{
			ComputeNode otherComputeNode=(ComputeNode)otherNode;
			otherComputeNode.removeInputNode(toRemove);
			otherComputeNode.removeOutputNode(toRemove);
		}
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
	
	public Hashtable<ComputeNode, Matrix> getOutput(Hashtable<ComputeNode, Matrix> inputs)
	{
		Hashtable<ComputeNode, Matrix> outputs=new Hashtable<>();
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> allOutputs=compute(inputs);
		for(ComputeNode outputNode: outputVertices.keySet())
		{
			for(Edge outputNodeEdge: outputNode.getEdgeSet())
			{
				if(!outputNodeEdge.getTargetNode().equals(outputNode))
				{
					outputs.put(outputNode, allOutputs.get(outputNodeEdge.getTargetNode()).get(outputNode));
					break;
				}
			}
		}
		return outputs;
	}
	
	//Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>>, Hashtable<ComputeNode, Matrix>
	public Object[] compute(Hashtable<ComputeNode, Matrix> inputs)
	{
		if(computeOrder==null)
		{
			computeOrder=determineComputeOrder();
		}
		
		Hashtable<ComputeNode, Matrix> allOutputs=new Hashtable<>();
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
			if(toCompute.getId().equals("outputTanH"))
			{
				int u=0;
			}
			if((allInputs.get(toCompute)!=null && allInputs.get(toCompute).size()==toCompute.inputNodes.length) 
					|| toCompute.inputNodes.length==0)
			{
				Hashtable<ComputeNode, Matrix> nodeOutputs=toCompute.getOutput(allInputs.get(toCompute));
				for(ComputeNode nodeOutputTo: nodeOutputs.keySet())
				{
					allOutputs.put(toCompute, nodeOutputs.get(nodeOutputTo));
					if(allInputs.get(nodeOutputTo)==null)
					{
						allInputs.put(nodeOutputTo, new Hashtable<ComputeNode, Matrix>());
					}
					allInputs.get(nodeOutputTo).put(toCompute, nodeOutputs.get(nodeOutputTo));
				}
			}
		}
		return new Object[]{allInputs, allOutputs};
	}
	
	public Hashtable<ComputeNode, Matrix> derive(Hashtable<ComputeNode, Matrix> inputs)
	{
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> allInputs
			=(Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>>)compute(inputs)[0];
		
		Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> objectiveDerivatives=new Hashtable<>();
		Hashtable<ComputeNode, Matrix> parameterDerivatives=new Hashtable<>();
		
		for(int computeInd=computeOrder.size()-1; computeInd>=0; computeInd--)
		{
			ComputeNode nextComputeNode=computeOrder.get(computeInd);
			if(nextComputeNode.getFunction() instanceof DifferentiableFunction
					&& (allInputs.get(nextComputeNode)!=null || nextComputeNode.inputNodes.length==0) 
					&& (objectiveDerivatives.get(nextComputeNode)!=null
					|| nextComputeNode.outputNodes.length==1 &&
					nextComputeNode.outputNodes[0].equals(nextComputeNode)))
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
		Hashtable<ComputeNode, ComputeNode> nodesToComputeTo=new Hashtable<>();
		List<ComputeNode> remainingNodes=new ArrayList<>(getNodeSet());
		
		for(ComputeNode outputNode: outputVertices.keySet())
		{
			nodesToComputeTo.put(outputNode, outputNode);
			addedNode.put(outputNode, outputNode);
		}
		
		while(!nodesToComputeTo.isEmpty())
		{
			ComputeNode computeNode=remainingNodes.remove(0);
			
			boolean allinputsComputed=true;
			for(ComputeNode inputNode: computeNode.inputNodes)
			{
				if(addedNode.get(inputNode)==null
						&& !inputNode.equals(computeNode))
				{
					allinputsComputed=false;
				}
			}
			
			if(allinputsComputed)
			{
				newComputeOrder.add(computeNode);
				addedNode.put(computeNode, computeNode);
				if(nodesToComputeTo.get(computeNode)!=null)
				{
					nodesToComputeTo.remove(computeNode);
				}
			}
			else
			{
				remainingNodes.add(computeNode);
			}
		}

		return newComputeOrder;
	}

}
