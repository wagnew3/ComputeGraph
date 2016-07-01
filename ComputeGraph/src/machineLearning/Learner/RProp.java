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
	Hashtable<String, Matrix> previousDerivatives;
	Hashtable<String, Matrix> deltasChanges;
	Hashtable<String, Matrix> deltas;
	
	float np=1.2f;
	float nm=0.5f;
	float maxDelta=50.0f;
	float minDelta=0.000001f;

	public RProp(List<Hashtable<String, Matrix>> examples, 
			List<Hashtable<String, Matrix>> validationExamples,
			int batchSize, int numberEpochs) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		initialized=false;
		previousDerivatives=new Hashtable<>();
		deltasChanges=new Hashtable<>();
		deltas=new Hashtable<>();
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<String, Matrix> batchDerivatives)
	{
		if(!initialized)
		{
			for(String cNode: batchDerivatives.keySet())
			{
				previousDerivatives.put(cNode, new FMatrix(batchDerivatives.get(cNode).getRows(), batchDerivatives.get(cNode).getCols()));
				FMatrix deltasChangesMatrix=new FMatrix(batchDerivatives.get(cNode).getRows(), batchDerivatives.get(cNode).getCols());
				for(int rowInd=0; rowInd<deltasChangesMatrix.getRows(); rowInd++)
				{
					for(int colInd=0; colInd<deltasChangesMatrix.getCols(); colInd++)
					{
						deltasChangesMatrix.set(rowInd, colInd, 0.1f);
					}
				}
				deltasChanges.put(cNode, deltasChangesMatrix);
				deltas.put(cNode, new FMatrix(batchDerivatives.get(cNode).getRows(), batchDerivatives.get(cNode).getCols()));
			}
			initialized=true;
		}
		
		float batchSizeScale=1.0f/getBatchSize();
		for(String vertex: batchDerivatives.keySet())
		{
			Matrix curDers=batchDerivatives.get(vertex);
			curDers.omscal(batchSizeScale);
			Matrix prevDers=previousDerivatives.get(vertex);
			Matrix deltasChange=deltasChanges.get(vertex);
			Matrix delta=deltas.get(vertex);
			
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
					}
					else
					{
						delta.set(devRow, devCol, Math.signum(curDers.get(devRow, devCol))*deltasChange.get(devRow, devCol));
					}
				}
			}
			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)((ComputeNode)cg.getNode(vertex)).getFunction());
			updateFunction.updateParameter(updateFunction.getParameter().omsub(delta));
			previousDerivatives.put(vertex, curDers);
		}
	}

	@Override
	public float validate(ComputeGraph cg, List<Hashtable<String, Matrix>> validationExamples) 
	{
		float totalError=0.0f;
		for(Hashtable<String, Matrix> validationExample: validationExamples)
		{
			Hashtable<String, Matrix> output=cg.getOutput(validationExample);
			totalError+=output.get("cost function").get(0, 0);
		}
		totalError/=validationExamples.size();
		return totalError;
	}

}
