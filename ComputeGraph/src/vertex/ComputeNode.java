package vertex;

import java.util.Hashtable;

import org.graphstream.graph.implementations.AbstractGraph;
import org.graphstream.graph.implementations.SingleNode;

import function.Function;
import matrix.Matrix;

public class ComputeNode extends SingleNode 
{

	Function function;
	
	public ComputeNode(AbstractGraph graph, String id)
	{
		super(graph, id);
	}
	
	public Matrix getOutput(Hashtable<String, Matrix> input)
	{
		return function.apply(input);
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
