package machineLearning.matrixTransform;

import function.DifferentiableFunction;
import function.UpdatableDifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class EbeMult extends DifferentiableFunction 
{

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix mat0Deriv=new FMatrix(input[0].getRows(), input[0].getCols());
		Matrix mat1Deriv=new FMatrix(input[1].getRows(), input[1].getCols());
		
		mat0Deriv=input[1].ebemult(dInput[0], mat0Deriv);
		mat1Deriv=input[0].ebemult(dInput[0], mat1Deriv);
		
		return new Matrix[][]{new Matrix[]{mat0Deriv, mat1Deriv}, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		Matrix ebeMulted=new FMatrix(input[0].getRows(), input[0].getCols());
		ebeMulted=input[0].ebemult(input[1], ebeMulted);
		return new Matrix[]{ebeMulted};
	}

}
