package machineLearning.generalFunctions;

import java.util.Hashtable;

import function.Function;
import matrix.FMatrix;
import matrix.Matrix;

public class Input extends Function
{

	@Override
	public Matrix apply(Hashtable<String, Matrix> input) 
	{
		return input.get("in");
	}

}
