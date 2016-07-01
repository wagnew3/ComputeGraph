package machineLearning.matrixTransform;

import function.DifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Combine extends DifferentiableFunction
{
	
	int combineIndex;
	int combineDimension;
	
	public Combine(int combineIndex, int combineDimension)
	{
		this.combineIndex=combineIndex;
		this.combineDimension=combineDimension;
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix split0=null;
		Matrix split1=null;
		if(combineDimension==0)
		{
			split0=new FMatrix(combineIndex, dInput[0].getCols());
			split1=new FMatrix(dInput[0].getRows()-combineIndex, dInput[0].getCols());
			for(int unsplitRowInd=0; unsplitRowInd<split0.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split0.getCols(); unsplitColInd++)
				{
					split0.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=0; unsplitRowInd<split1.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split1.getCols(); unsplitColInd++)
				{
					split1.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd+combineIndex, unsplitColInd));
				}
			}
		}
		else if(combineDimension==1)
		{
			split0=new FMatrix(dInput[0].getRows(), combineIndex);
			split1=new FMatrix(dInput[0].getRows(), dInput[0].getCols()-combineIndex);
			for(int unsplitRowInd=0; unsplitRowInd<split0.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split0.getCols(); unsplitColInd++)
				{
					split0.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=0; unsplitRowInd<split1.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split1.getCols(); unsplitColInd++)
				{
					split1.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd+combineIndex));
				}
			}
		}
		return new Matrix[][]{new Matrix[]{split0, split1}, null};	
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		Matrix combined=null;
		if(combineDimension==0)
		{
			combined=new FMatrix(input[0].getRows()+input[1].getRows(), input[0].getCols());
			for(int unsplitRowInd=0; unsplitRowInd<input[0].getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<input[0].getCols(); unsplitColInd++)
				{
					combined.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=input[0].getRows(); unsplitRowInd<input[0].getRows()+input[1].getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<input[1].getCols(); unsplitColInd++)
				{
					combined.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd-input[0].getRows(), unsplitColInd));
				}
			}
		}
		else if(combineDimension==1)
		{
			combined=new FMatrix(input[0].getRows(), input[0].getCols()+input[1].getCols());
			for(int unsplitRowInd=0; unsplitRowInd<input[0].getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<input[0].getCols(); unsplitColInd++)
				{
					combined.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=0; unsplitRowInd<input[1].getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=input[0].getCols(); unsplitColInd<input[0].getCols()+input[1].getCols(); unsplitColInd++)
				{
					combined.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd-input[0].getCols()));
				}
			}
		}
		return new Matrix[]{combined};
	}

}
