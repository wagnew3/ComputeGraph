package machineLearning.matrixTransform;

import java.util.Hashtable;

import function.UpdatableDifferentiableFunction;
import matrix.Matrix;

public class MAdd extends UpdatableDifferentiableFunction
{
	
	Matrix add;
	
	public MAdd(Matrix add)
	{
		this.add=add;
	}

	@Override
	public Matrix getParameter() 
	{
		return add;
	}

	@Override
	public void updateParameter(Matrix newValue)
	{
		add=newValue;
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix derivative=dInput[0];
		return new Matrix[][]{new Matrix[]{derivative}, new Matrix[]{derivative}};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		Matrix inputMatrix=input[0];
		return new Matrix[]{inputMatrix.mad(add)};
	}

}
