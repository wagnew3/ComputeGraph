package machineLearning.generalFunctions;

import function.UpdatableDifferentiableFunction;
import matrix.Matrix;

public class Constant extends UpdatableDifferentiableFunction
{
	
	Matrix output;
	
	public Constant(Matrix output)
	{
		this.output=output;
	}

	@Override
	public Matrix getParameter()
	{
		return output;
	}

	@Override
	public void updateParameter(Matrix newValue)
	{
		output=newValue;
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput)
	{
		return new Matrix[][]{null, new Matrix[]{dInput[0]}};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		return new Matrix[]{output};
	}

}
