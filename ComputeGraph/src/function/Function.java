package function;

import java.util.Hashtable;

import matrix.Matrix;
import vertex.ComputeNode;

public abstract class Function 
{
	protected static String[] expectedInputs;
	
	public abstract Matrix[] apply(Matrix[] input);

}
