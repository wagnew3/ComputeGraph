package function;

import matrix.Matrix;

public abstract class UpdatableDifferentiableFunction extends DifferentialbleFunction
{
	
	public abstract Matrix getParameter();
	
	public abstract void updateParameter(Matrix newValue);

}
