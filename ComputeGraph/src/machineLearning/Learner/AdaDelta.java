package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import graph.ComputeGraph;
import matrix.Matrix;
import vertex.ComputeNode;

public class AdaDelta extends ExampleBatchDerivativeOptimizer
{
	
	Hashtable<ComputeNode, Matrix> squaredWeightedGradientAverage;
	float averageWeight=

	public AdaDelta(List<Hashtable<ComputeNode, Matrix>> examples,
			List<Hashtable<ComputeNode, Matrix>> validationExamples, 
			int batchSize, int numberEpochs) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		squaredWeightedGradientAverage=new Hashtable<ComputeNode, Matrix>();
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<ComputeNode, Matrix> batchDerivatives) 
	{
		
	}

	@Override
	public float validate(ComputeGraph cg, List<Hashtable<ComputeNode, Matrix>> validationExamples,
			ComputeNode[] objectives) 
	{

		return 0;
	}

}
