package machineLearning.matrixTransform;

import java.util.Hashtable;

import function.UpdatableDifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class MMult extends UpdatableDifferentiableFunction
{
	
	Matrix paramMatrix;
	
	public MMult(Matrix paramMatrix)
	{
		this.paramMatrix=paramMatrix;
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix paramDiff=new FMatrix(input[0].getRows(), dInput[0].getRows());
		dInput[0].outProd(dInput[0], input[0], paramDiff);
		Matrix objectiveDiff=new FMatrix(input[0].getRows(), 1);
		objectiveDiff=((FMatrix)objectiveDiff).sgemv(true, 1.0f, paramMatrix, 
				dInput[0], 1, 1.0f, objectiveDiff, 1, true, objectiveDiff);
		
		return new Matrix[][]{new Matrix[]{objectiveDiff}, new Matrix[]{paramDiff}};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		return new Matrix[]{paramMatrix.mmult(input[0])};
	}

	@Override
	public Matrix getParameter() 
	{
		return paramMatrix;
	}

	@Override
	public void updateParameter(Matrix newValue) 
	{
		paramMatrix=newValue;
	}

}
