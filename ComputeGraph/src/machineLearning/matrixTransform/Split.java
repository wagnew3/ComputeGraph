package machineLearning.matrixTransform;

import function.DifferentiableFunction;
import matrix.FMatrix;
import matrix.Matrix;

public class Split extends DifferentiableFunction
{
	
	int splitIndex;
	int splitDimension;
	int combinedSize;
	
	public Split(int splitIndex, int splitDimension, int combinedSize)
	{
		this.splitIndex=splitIndex;
		this.splitDimension=splitDimension;
		this.combinedSize=combinedSize;
	}

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix dUnsplit=null;
		
		if(dInput.length==1 || dInput[1]==null)
		{
			dUnsplit=new FMatrix(combinedSize, dInput[0].getCols());
			for(int unsplitRowInd=0; unsplitRowInd<dInput[0].getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<dInput[0].getCols(); unsplitColInd++)
				{
					dUnsplit.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd));
				}
			}
		}
		else
		{
			if(splitDimension==0)
			{
				dUnsplit=new FMatrix(dInput[0].getRows()+dInput[1].getRows(), dInput[0].getCols());
				for(int unsplitRowInd=0; unsplitRowInd<dInput[0].getRows(); unsplitRowInd++)
				{
					for(int unsplitColInd=0; unsplitColInd<dInput[0].getCols(); unsplitColInd++)
					{
						dUnsplit.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd));
					}
				}
				for(int unsplitRowInd=dInput[0].getRows(); unsplitRowInd<dInput[0].getRows()+dInput[1].getRows(); unsplitRowInd++)
				{
					for(int unsplitColInd=0; unsplitColInd<dInput[1].getCols(); unsplitColInd++)
					{
						dUnsplit.set(unsplitRowInd, unsplitColInd, dInput[1].get(unsplitRowInd-dInput[0].getRows(), unsplitColInd));
					}
				}
			}
			else if(splitDimension==1)
			{
				dUnsplit=new FMatrix(dInput[0].getRows(), dInput[0].getCols()+dInput[1].getCols());
				for(int unsplitRowInd=0; unsplitRowInd<dInput[0].getRows(); unsplitRowInd++)
				{
					for(int unsplitColInd=0; unsplitColInd<dInput[0].getCols(); unsplitColInd++)
					{
						dUnsplit.set(unsplitRowInd, unsplitColInd, dInput[0].get(unsplitRowInd, unsplitColInd));
					}
				}
				for(int unsplitRowInd=0; unsplitRowInd<dInput[1].getRows(); unsplitRowInd++)
				{
					for(int unsplitColInd=dInput[0].getCols(); unsplitColInd<dInput[0].getCols()+dInput[1].getCols(); unsplitColInd++)
					{
						dUnsplit.set(unsplitRowInd, unsplitColInd, dInput[1].get(unsplitRowInd, unsplitColInd-dInput[0].getCols()));
					}
				}
			}
		}
		return new Matrix[][]{new Matrix[]{dUnsplit}, null};
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		Matrix split0=null;
		Matrix split1=null;
		if(splitDimension==0)
		{
			split0=new FMatrix(splitIndex, input[0].getCols());
			split1=new FMatrix(input[0].getRows()-splitIndex, input[0].getCols());
			for(int unsplitRowInd=0; unsplitRowInd<split0.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split0.getCols(); unsplitColInd++)
				{
					split0.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=0; unsplitRowInd<split1.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split1.getCols(); unsplitColInd++)
				{
					split1.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd+splitIndex, unsplitColInd));
				}
			}
		}
		else if(splitDimension==1)
		{
			split0=new FMatrix(input[0].getRows(), splitIndex);
			split1=new FMatrix(input[1].getRows(), input[1].getCols()-splitIndex);
			for(int unsplitRowInd=0; unsplitRowInd<split0.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split0.getCols(); unsplitColInd++)
				{
					split0.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd));
				}
			}
			for(int unsplitRowInd=0; unsplitRowInd<split1.getRows(); unsplitRowInd++)
			{
				for(int unsplitColInd=0; unsplitColInd<split1.getCols(); unsplitColInd++)
				{
					split1.set(unsplitRowInd, unsplitColInd, input[0].get(unsplitRowInd, unsplitColInd+splitIndex));
				}
			}
		}
		return new Matrix[]{split0, split1};
	}

}
