package function;

import java.util.Hashtable;

import matrix.Matrix;

public abstract class Function 
{
	protected static String[] expectedInputs;
	
	public abstract Matrix apply(Hashtable<String, Matrix> input);

}
