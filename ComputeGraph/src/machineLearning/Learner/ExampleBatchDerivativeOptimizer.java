package machineLearning.Learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import graph.ComputeGraph;
import matrix.Matrix;
import optimizer.Optimizer;
import vertex.ComputeNode;

public abstract class ExampleBatchDerivativeOptimizer extends Optimizer
{
	
	private List<Hashtable<ComputeNode, Matrix>> examples;
	private List<Hashtable<ComputeNode, Matrix>> validationExamples;
	private int numberEpochs;
	private int batchSize;
	
	public ExampleBatchDerivativeOptimizer(List<Hashtable<ComputeNode, Matrix>> examples, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples,
			int batchSize,
			int numberEpochs)
	{
		this.examples=examples;
		this.validationExamples=validationExamples;
		this.numberEpochs=numberEpochs;
		this.batchSize=batchSize;
	}

	@Override
	public void optimize(ComputeGraph cg, ComputeNode objective) 
	{
		List<Hashtable<ComputeNode, Matrix>> randomizedExamples=new ArrayList<>();
		randomizedExamples.addAll(examples);
		for(int epoch=0; epoch<numberEpochs; epoch++)
		{
			//Collections.shuffle(randomizedExamples);
			System.out.println("unrandomized inputs");
						
			for(int sampleInd=0; sampleInd<randomizedExamples.size(); sampleInd+=batchSize)
			{
				Hashtable<ComputeNode, Matrix> batchParameterDerivatives=new Hashtable<>();
				Hashtable<ComputeNode, Matrix>[] batchExamples=new Hashtable[Math.min(batchSize, randomizedExamples.size()-sampleInd)];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					batchExamples[batchInd]=randomizedExamples.get(batchInd+sampleInd);
				}
				
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					Hashtable<ComputeNode, Matrix> parameterDerivatives=cg.derive(batchExamples[batchInd]);
					if(batchInd==0)
					{
						for(ComputeNode nodeDeriv: parameterDerivatives.keySet())
						{
							batchParameterDerivatives.put(nodeDeriv, parameterDerivatives.get(nodeDeriv));
						}
					}
					else
					{
						for(ComputeNode nodeDeriv: parameterDerivatives.keySet())
						{
							batchParameterDerivatives.get(nodeDeriv).omad(parameterDerivatives.get(nodeDeriv));
						}
					}
				}
				updateParameters(cg, batchParameterDerivatives);
			}
			float validationError=validate(cg, validationExamples, objective);
			System.out.println("Epoch: "+epoch+" Average Validation Error: "+validationError);
		}
	}
	
	public int getNumberEpochs()
	{
		return numberEpochs;
	}
	
	public int getBatchSize()
	{
		return batchSize;
	}
	
	public abstract void updateParameters(ComputeGraph cg, 
			Hashtable<ComputeNode, Matrix> batchDerivatives);
	
	public abstract float validate(ComputeGraph cg, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples, ComputeNode objective);

}
