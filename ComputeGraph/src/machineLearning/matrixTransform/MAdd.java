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
	public Matrix[] differentiate(Hashtable<String, Matrix> input, Hashtable<String, Matrix> dInput) 
	{
		Matrix derivative=dInput.get("in");
		return new Matrix[]{derivative, derivative};
	}

	@Override
	public Matrix apply(Hashtable<String, Matrix> input) 
	{
		Matrix inputMatrix=input.get("in");
		return inputMatrix.mad(add);
	}

}
