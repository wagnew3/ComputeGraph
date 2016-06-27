package machineLearning.costFunction;

import java.util.Hashtable;

import function.DifferentialbleFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Euclidean extends DifferentialbleFunction
{
	
	static
	{
		expectedInputs=new String[]{"network", "train"};
	}

	@Override
	public Matrix[] differentiate(Hashtable<String, Matrix> input, Hashtable<String, Matrix> dInput)
	{
		Matrix networkOutput=input.get("network");
		Matrix trainOutputs=input.get("train");
		Matrix result=new FMatrix(trainOutputs.getLen(), 1);
		
		result=networkOutput.msub(trainOutputs, result);
		result.omscal(2.0f/trainOutputs.getLen());
		
		return new Matrix[]{result, null};
	}

	@Override
	public Matrix apply(Hashtable<String, Matrix> input)
	{
		Matrix networkOutput=input.get("network");
		Matrix trainOutputs=input.get("train");
		
		float total=0.0f;
		Matrix difference=new FMatrix(trainOutputs.getLen(), 1);
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
			difference=networkOutput.msub(trainOutputs, difference);
		}
		total+=(float)(0.5*difference.dot(difference));
		difference.clear();
		Matrix result=new FMatrix(1,1);
		result.set(0, 0, total);
		return result;
	}

}
