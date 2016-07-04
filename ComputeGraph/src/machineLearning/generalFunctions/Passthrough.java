package machineLearning.generalFunctions;

import function.DifferentiableFunction;
import matrix.Matrix;

public class Passthrough extends DifferentiableFunction
{

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		return new Matrix[][]{dInput, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		return input;
	}

}
