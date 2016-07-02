package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import function.UpdatableDifferentiableFunction;
import graph.ComputeGraph;
import matrix.Matrix;
import vertex.ComputeNode;

public class BackPropagation extends ExampleBatchDerivativeOptimizer
{

	protected float learningRate;
	
	public BackPropagation(List<Hashtable<ComputeNode, Matrix>> examples, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples,
			int batchSize, int numberEpochs, float learningRate) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		this.learningRate=learningRate;
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<ComputeNode, Matrix> batchDerivatives)
	{
		float batchSizeScale=1.0f/getBatchSize();
		batchSizeScale*=learningRate;
		for(ComputeNode vertex: batchDerivatives.keySet())
		{
			Matrix update=batchDerivatives.get(vertex);
			update.omscal(batchSizeScale);
			
			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)(vertex).getFunction());
			
			update=updateFunction.getParameter().omsub(update);
			updateFunction.updateParameter(update);
		}
	}

	@Override
	public float validate(ComputeGraph cg, List<Hashtable<ComputeNode, Matrix>> validationExamples,
			ComputeNode[] objectives) 
	{
		float totalError=0.0f;
		
		
		int numberCorrect=0;
		long numberErrors=0;
		for(int objectiveInd=0; objectiveInd<objectives.length; objectiveInd++)
		{
			for(Hashtable<ComputeNode, Matrix> validationExample: validationExamples)
			{
				Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> output=cg.compute(validationExample);
				if(output.get(objectives[objectiveInd])!=null)
				{
					totalError+=output.get(objectives[objectiveInd]).get(objectives[objectiveInd]).get(0, 0);
					numberErrors++;
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
		}
		
		System.out.println("Number classified incorrectly: "+numberCorrect);
		
		totalError/=numberErrors;
		return totalError;
	}

}
