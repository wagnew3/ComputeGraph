package machineLearning.costFunction;

import java.util.Hashtable;

import function.DifferentialbleFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Euclidean extends DifferentialbleFunction
{
	
	static
	{
		expectedInputs=new String[]{"network output", "train output"};
	}

	@Override
	public Matrix[] differentiate(Hashtable<String, Matrix> input, Hashtable<String, Matrix> dInput)
	{
		Matrix[] networkOutput=new Matrix[input.size()/2];
		for(int inputInd=0; inputInd<networkOutput.length; inputInd++)
		{
			networkOutput[inputInd]=input.get("network output"+inputInd);
		}
		Matrix[] trainOutputs=new Matrix[input.size()/2];
		for(int inputInd=0; inputInd<trainOutputs.length; inputInd++)
		{
			trainOutputs[inputInd]=input.get("train output"+inputInd);
		}
		
		Matrix result=new FMatrix(trainOutputs[0].getLen(), 1);
		
		for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
		{
			result=result.omad(networkOutput[outputInd].omsub(trainOutputs[outputInd]));
		}
		return new Matrix[]{result, null};
	}

	@Override
	public Matrix apply(Hashtable<String, Matrix> input)
	{
		Matrix[] networkOutput=new Matrix[input.size()/2];
		for(int inputInd=0; inputInd<networkOutput.length; inputInd++)
		{
			networkOutput[inputInd]=input.get("network output"+inputInd);
		}
		Matrix[] trainOutputs=new Matrix[input.size()/2];
		for(int inputInd=0; inputInd<trainOutputs.length; inputInd++)
		{
			trainOutputs[inputInd]=input.get("train output"+inputInd);
		}
		
		float total=0.0f;
		for(int outputInd=0; outputInd<networkOutput.length; outputInd++)
		{
			Matrix difference=null;
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
				difference=networkOutput[outputInd].msub(trainOutputs[outputInd], networkOutput[outputInd]);
			}
			total+=(float)(0.5*difference.dot(difference));
			difference.clear();
		}
		Matrix result=new FMatrix(1,1);
		result.set(0, 0, total);
		return result;
	}

}
