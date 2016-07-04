package machineLearning.matrixTransform;

import function.DifferentiableFunction;
import matrix.Matrix;

public class MAdd extends DifferentiableFunction 
{

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		return new Matrix[][]{new Matrix[]{dInput[0], dInput[0]}, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		return new Matrix[]{input[0].mad(input[1])};
	}

}
