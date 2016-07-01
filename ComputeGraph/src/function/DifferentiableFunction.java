package function;

import java.util.Hashtable;

import matrix.Matrix;
import vertex.ComputeNode;

public abstract class DifferentiableFunction extends Function
{
	
	public abstract Matrix[][] differentiate(Matrix[] input,
			Matrix[] dInput);

}
