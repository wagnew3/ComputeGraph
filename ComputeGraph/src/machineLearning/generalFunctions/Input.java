package machineLearning.generalFunctions;

import java.util.Hashtable;

import function.DifferentialbleFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Input extends DifferentialbleFunction
{

	@Override
	public Matrix[] differentiate(Hashtable<String, Matrix> input, Hashtable<String, Matrix> dInput) 
	{
		Matrix derivative=new FMatrix(input.get("in").getLen(), 1);
		for(int inputDevInd=0; inputDevInd<derivative.getLen(); inputDevInd++)
		{
			derivative.set(inputDevInd, 0, 1.0f);
		}
		return new Matrix[]{derivative, null};
	}

	@Override
	public Matrix apply(Hashtable<String, Matrix> input) 
	{
		return input.get("in");
	}

}
