package vertex;

import java.util.Hashtable;

import org.graphstream.graph.implementations.AbstractGraph;
import org.graphstream.graph.implementations.SingleNode;

import function.DifferentiableFunction;
import function.Function;
import graph.ComputeGraph;
import matrix.Matrix;

public class ComputeNode extends SingleNode 
{

	Function function;
	public ComputeNode[] inputNodes;
	public ComputeNode[] outputNodes;
	
	public ComputeNode(AbstractGraph graph, String id)
	{
		super(graph, id);
	}
	
	public void setInputNode(ComputeNode[] setInputNodes)
	{
		this.inputNodes=setInputNodes;
		for(ComputeNode inputNode: inputNodes)
		{
			((ComputeGraph)graph).addEdge(inputNode, this);
		}
	}
	
	public void setOutputNode(ComputeNode[] setOutputNodes)
	{
		this.outputNodes=setOutputNodes;
		for(ComputeNode outputNode: outputNodes)
		{
			((ComputeGraph)graph).addEdge(this, outputNode);
		}
	}
	
	public void setInputOutputNode(ComputeNode[] setInputNodes, ComputeNode[] setOutputNodes)
	{
		setInputNode(setInputNodes);
		setOutputNode(setOutputNodes);
	}
	
	public Hashtable<ComputeNode, Matrix> getOutput(Hashtable<ComputeNode, Matrix> input)
	{
		Matrix[] inputs=new Matrix[inputNodes.length];
		for(int inputInd=0; inputInd<inputNodes.length; inputInd++)
		{
			inputs[inputInd]=input.get(inputNodes[inputInd]);
		}
		Matrix[] outputMatrices=function.apply(inputs);
		Hashtable<ComputeNode, Matrix> outputs=new Hashtable<>();
		for(int outputInd=0; outputInd<outputNodes.length; outputInd++)
		{
			outputs.put(outputNodes[outputInd], outputMatrices[outputInd]);
		}
		return outputs;
	}
	
	public Hashtable<ComputeNode, Matrix>[] differentiate(Hashtable<ComputeNode, Matrix> input,
			Hashtable<ComputeNode, Matrix> dInput)
	{
		Matrix[] inputs=new Matrix[inputNodes.length];
		for(int inputInd=0; inputInd<inputNodes.length; inputInd++)
		{
			inputs[inputInd]=input.get(inputNodes[inputInd]);
		}
		
		Matrix[] dInputs=new Matrix[outputNodes.length];
		for(int inputInd=0; inputInd<outputNodes.length; inputInd++)
		{
			if(dInput!=null)
			{
				dInputs[inputInd]=dInput.get(outputNodes[inputInd]);
			}
		}
		
		Matrix[][] outputMatrices=((DifferentiableFunction)function).differentiate(inputs, dInputs);
		Hashtable<ComputeNode, Matrix> objectiveDevs=new Hashtable<>();
		for(int outputInd=0; outputInd<inputNodes.length && outputInd<outputMatrices[0].length; outputInd++)
		{
			objectiveDevs.put(inputNodes[outputInd], outputMatrices[0][outputInd]);
		}
		Hashtable<ComputeNode, Matrix> paramDevs=new Hashtable<>();
		if(outputMatrices[1]!=null)
		{
			paramDevs.put(this, outputMatrices[1][0]);
		}

		return new Hashtable[]{objectiveDevs, paramDevs};
	}
	
	public Function getFunction()
	{
		return function;
	}
	
	public void setFunction(Function setFunction)
	{
		this.function=setFunction;
	}

}
