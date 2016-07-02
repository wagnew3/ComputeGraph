package machineLearning.costFunction;

import java.util.Hashtable;

import function.DifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Euclidean extends DifferentiableFunction
{
	
	static
	{
		expectedInputs=new String[]{"network", "train"};
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput)
	{
		Matrix[] derivatives=new Matrix[input.length/2];
		for(int outputNumber=0; outputNumber<input.length/2; outputNumber++)
		{
			Matrix result=new FMatrix(input[2*outputNumber+1].getLen(), 1);
			result=input[2*outputNumber].msub(input[2*outputNumber+1], result);
			derivatives[outputNumber/2]=result;
		}
		return new Matrix[][]{derivatives, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input)
	{
		float total=0.0f;
		for(int outputNumber=0; outputNumber<input.length/2; outputNumber++)
		{
			Matrix difference=new FMatrix(input[2*outputNumber+1].getLen(), 2*outputNumber+1);
			if(false) //SparseArrayRealVector
			{
				/*
				SparseArrayRealVector sparseDesiredOutput=(SparseArrayRealVector)desiredOutput;
				
				difference=new ArrayRealVector(sparseDesiredOutput.getDimension());
				
				difference=difference.add(networkOutput);
				for(int entryInd=0; entryInd<sparseDesiredOutput.nonZeroEntries.length; entryInd++)
				{
					difference.addToEntry((int)sparseDesiredOutput.nonZeroEntries[entryInd], 
							-1.0*sparseDesiredOutput.sparseData[entryInd]);
				}
				*/
			}
			else
			{
				difference=input[2*outputNumber].msub(input[2*outputNumber+1], difference);
			}
			total+=(float)(difference.dot(difference));
			total/=input[2*outputNumber+1].getLen();
			difference.clear();
		}
		Matrix result=new FMatrix(1,1);
		result.set(0, 0, total);
		return new Matrix[]{result};
	}

}
