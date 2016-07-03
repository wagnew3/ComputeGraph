package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import function.UpdatableDifferentiableFunction;
import graph.ComputeGraph;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class RProp extends ExampleBatchDerivativeOptimizer
{
	
	boolean initialized;
	Hashtable<ComputeNode, Matrix> previousDerivatives;
	Hashtable<ComputeNode, Matrix> deltasChanges;
	Hashtable<ComputeNode, Matrix> deltas;
	
	float np=1.1f;
	float nm=0.05f;
	float maxDelta=50.0f;
	float minDelta=0.000001f;

	public RProp(List<Hashtable<ComputeNode, Matrix>> examples, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples,
			int batchSize, int numberEpochs) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		initialized=false;
		previousDerivatives=new Hashtable<>();
		deltasChanges=new Hashtable<>();
		deltas=new Hashtable<>();
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<ComputeNode, Matrix> batchDerivatives)
	{
		if(!initialized)
		{
			for(ComputeNode node: batchDerivatives.keySet())
			{
				previousDerivatives.put(node, new FMatrix(batchDerivatives.get(node).getRows(), batchDerivatives.get(node).getCols()));
				FMatrix deltasChangesMatrix=new FMatrix(batchDerivatives.get(node).getRows(), batchDerivatives.get(node).getCols());
				for(int rowInd=0; rowInd<deltasChangesMatrix.getRows(); rowInd++)
				{
					for(int colInd=0; colInd<deltasChangesMatrix.getCols(); colInd++)
					{
						deltasChangesMatrix.set(rowInd, colInd, 0.01f);
					}
				}
				deltasChanges.put(node, deltasChangesMatrix);
				deltas.put(node, new FMatrix(batchDerivatives.get(node).getRows(), batchDerivatives.get(node).getCols()));
			}
			initialized=true;
		}
		
		for(ComputeNode node: batchDerivatives.keySet())
		{
			Matrix curDers=batchDerivatives.get(node);
			Matrix prevDers=previousDerivatives.get(node);
			Matrix deltasChange=deltasChanges.get(node);
			Matrix delta=deltas.get(node);
			
			for(int devRow=0; devRow<prevDers.getRows(); devRow++)
			{
				for(int devCol=0; devCol<prevDers.getCols(); devCol++)
				{
					float devProd=Math.signum(prevDers.get(devRow, devCol))*Math.signum(curDers.get(devRow, devCol));
					if(devProd>0)
					{
						deltasChange.set(devRow, devCol, Math.min(deltasChange.get(devRow, devCol)*np, maxDelta));
						delta.set(devRow, devCol, Math.signum(curDers.get(devRow, devCol))*deltasChange.get(devRow, devCol));
					}
					else if(devProd<0)
					{
						deltasChange.set(devRow, devCol, Math.max(deltasChange.get(devRow, devCol)*nm, minDelta));
						delta.set(devRow, devCol, -1.0f*delta.get(devRow, devCol));
						curDers.set(devRow, devCol, 0.0f);
					}
					else
					{
						delta.set(devRow, devCol, Math.signum(curDers.get(devRow, devCol))*deltasChange.get(devRow, devCol));
					}
				}
			}
			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)node.getFunction());
			updateFunction.updateParameter(updateFunction.getParameter().omsub(delta));
			previousDerivatives.put(node, curDers);
		}
	}

	@Override
	public float validate(ComputeGraph cg, List<Hashtable<ComputeNode, Matrix>> validationExamples,
			ComputeNode[] objectives) 
	{
		float totalError=0.0f;
	
		long numberErrors=0;
		double numberCorrect=0.0;
		double totalTrys=0.0;
		for(Hashtable<ComputeNode, Matrix> validationExample: validationExamples)
		{
			Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> output=cg.compute(validationExample);
			for(int objectiveInd=0; objectiveInd<objectives.length; objectiveInd++)
			{
				if(output.get(objectives[objectiveInd])!=null)
				{
					totalError+=output.get(objectives[objectiveInd]).get(objectives[objectiveInd]).get(0, 0);
					numberErrors++;
					
					Matrix netOut=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]);
					float netGuess=netOut.get(0, 0);
					
					Matrix trainOut=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[1]);
					float trainAnswer=trainOut.get(0, 0);
					
					netGuess=Math.round(netGuess);
					if(netGuess==trainAnswer)
					{
						numberCorrect++;
					}
					else
					{
						int u=0;
					}
					totalTrys++;
					
				}
			}
			/*
			float max=-1;
			int maxInd=-1;
			for(int outInd=0; outInd<output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).getLen(); outInd++)
			{
				if(max<output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).get(outInd, 0))
				{
					max=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).get(outInd, 0);
					maxInd=outInd;
				}
			}
			int correctInd=-2;
			for(int outInd=0; outInd<validationExample.get(objectives[objectiveInd].inputNodes[1]).getLen(); outInd++)
			{
				if(validationExample.get(objectives[objectiveInd].inputNodes[1]).get(outInd, 0)==1.0f)
				{
					correctInd=outInd;
					break;
				}
			}
			if(maxInd!=correctInd)
			{
				numberCorrect++;
			}
			*/
		}
		
		System.out.println("Number classified correctly: "+numberCorrect+"/"+totalTrys);
		
		totalError/=numberErrors;
		return totalError;
	}

}
