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
			ComputeNode objective) 
	{
		float totalError=0.0f;
		
		
		int numberCorrect=0;
		
		for(Hashtable<ComputeNode, Matrix> validationExample: validationExamples)
		{
			Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> output=cg.compute(validationExample);
			totalError+=output.get(objective).get(objective).get(0, 0);
			
			float max=-1;
			int maxInd=-1;
			for(int outInd=0; outInd<output.get(objective).get(objective.inputNodes[0]).getLen(); outInd++)
			{
				if(max<output.get(objective).get(objective.inputNodes[0]).get(outInd, 0))
				{
					max=output.get(objective).get(objective.inputNodes[0]).get(outInd, 0);
					maxInd=outInd;
				}
			}
			int correctInd=-2;
			for(int outInd=0; outInd<validationExample.get(objective.inputNodes[1]).getLen(); outInd++)
			{
				if(validationExample.get(objective.inputNodes[1]).get(outInd, 0)==1.0f)
				{
					correctInd=outInd;
					break;
				}
			}
			if(maxInd!=correctInd)
			{
				numberCorrect++;
			}
		
		}
		
		System.out.println("Number classified incorrectly: "+numberCorrect);
		
		totalError/=validationExamples.size();
		return totalError;
	}

}
