package function;

import java.util.Hashtable;

import matrix.Matrix;

public abstract class DifferentialbleFunction extends Function
{
	
	public abstract Matrix[] differentiate(Hashtable<String, Matrix> input,
			Hashtable<String, Matrix> dInput);

}
