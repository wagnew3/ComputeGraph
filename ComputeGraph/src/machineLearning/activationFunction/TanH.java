package machineLearning.activationFunction;

import java.util.Hashtable;

import function.DifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class TanH extends DifferentiableFunction
{

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix activation=apply(input)[0];
		activation=activation.oebemult(activation);
		
		for(int inputInd=0; inputInd<activation.getLen(); inputInd++)
		{
			activation.set(inputInd, 0, 1.0f-activation.get(inputInd, 0));
		}
		activation.oebemult(dInput[0]);
		return new Matrix[][]{new Matrix[]{activation}, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		Matrix output=new FMatrix(input[0].getRows(), input[0].getCols());
		for(int inputInd=0; inputInd<input[0].getLen(); inputInd++)
		{
			output.set(inputInd, 0, (float)Math.tanh(input[0].get(inputInd, 0)));
		}
		return new Matrix[]{output};
	}

}
