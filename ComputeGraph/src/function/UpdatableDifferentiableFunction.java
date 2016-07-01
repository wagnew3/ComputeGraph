package function;

import matrix.Matrix;

public abstract class UpdatableDifferentiableFunction extends DifferentiableFunction
{
	
	public abstract Matrix getParameter();
	
	public abstract void updateParameter(Matrix newValue);

}
