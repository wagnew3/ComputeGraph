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
	public Matrix[] differentiate(Hashtable<String, Matrix> input, Hashtable<String, Matrix> dInput) 
	{
		Matrix paramDiff=new FMatrix(input.get("in").getRows(), dInput.get("in").getRows());
		dInput.get("in").outProd(dInput.get("in"), input.get("in"), paramDiff);
		Matrix objectiveDiff=new FMatrix(input.get("in").getRows(), 1);
		objectiveDiff=paramMatrix.sgemv(true, 1.0f, paramMatrix, 
				dInput.get("in"), 1, 0.0f, objectiveDiff, 1, false, objectiveDiff);
		return new Matrix[]{objectiveDiff, paramDiff};
	}

	@Override
	public Matrix apply(Hashtable<String, Matrix> input) 
	{
		return paramMatrix.mmult(input.get("in"));
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
