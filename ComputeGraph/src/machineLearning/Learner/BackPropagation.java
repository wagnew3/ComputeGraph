package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import function.UpdatableDifferentiableFunction;
import graph.ComputeGraph;
import matrix.Matrix;
import vertex.ComputeNode;

public class BackPropagation extends ExampleBatchDerivativeOptimizer
{

	public BackPropagation(List<Hashtable<String, Matrix>> examples, 
			List<Hashtable<String, Matrix>> validationExamples,
			int batchSize, int numberEpochs) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<String, Matrix> batchDerivatives)
	{
		float batchSizeScale=1.0f/getBatchSize();
		for(String vertex: batchDerivatives.keySet())
		{
			Matrix update=batchDerivatives.get(vertex);
			update.omscal(batchSizeScale);
			
			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)((ComputeNode)cg.getNode(vertex)).getFunction());
			
			update=updateFunction.getParameter().omsub(update);
			updateFunction.updateParameter(update);
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
