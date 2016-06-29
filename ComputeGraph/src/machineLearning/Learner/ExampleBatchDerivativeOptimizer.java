package machineLearning.Learner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Hashtable;
import java.util.List;

import graph.ComputeGraph;
import matrix.Matrix;
import optimizer.Optimizer;

public abstract class ExampleBatchDerivativeOptimizer extends Optimizer
{
	
	private List<Hashtable<String, Matrix>> examples;
	private List<Hashtable<String, Matrix>> validationExamples;
	private int numberEpochs;
	private int batchSize;
	
	public ExampleBatchDerivativeOptimizer(List<Hashtable<String, Matrix>> examples, 
			List<Hashtable<String, Matrix>> validationExamples,
			int batchSize,
			int numberEpochs)
	{
		this.examples=examples;
		this.validationExamples=validationExamples;
		this.numberEpochs=numberEpochs;
		this.batchSize=batchSize;
	}

	@Override
	public void optimize(ComputeGraph cg, String objective) 
	{
		List<Hashtable<String, Matrix>> randomizedExamples=new ArrayList<>();
		randomizedExamples.addAll(examples);
		for(int epoch=0; epoch<numberEpochs; epoch++)
		{
			Collections.shuffle(randomizedExamples);
			//System.out.println("unrandomized inputs");
						
			for(int sampleInd=0; sampleInd<randomizedExamples.size(); sampleInd+=batchSize)
			{
				Hashtable<String, Matrix> batchParameterDerivatives=new Hashtable<>();
				Hashtable<String, Matrix>[] batchExamples=new Hashtable[Math.min(batchSize, randomizedExamples.size()-sampleInd)];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					batchExamples[batchInd]=randomizedExamples.get(batchInd+sampleInd);
				}
				
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					Hashtable<String, Matrix> parameterDerivatives=cg.derive(batchExamples[batchInd])[1];
					if(batchInd==0)
					{
						for(String nodeDeriv: parameterDerivatives.keySet())
						{
							batchParameterDerivatives.put(nodeDeriv, parameterDerivatives.get(nodeDeriv));
						}
					}
					else
					{
						for(String nodeDeriv: parameterDerivatives.keySet())
						{
							batchParameterDerivatives.get(nodeDeriv).omad(parameterDerivatives.get(nodeDeriv));
						}
					}
				}
				updateParameters(cg, batchParameterDerivatives);
			}
			float validationError=validate(cg, validationExamples);
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
			Hashtable<String, Matrix> batchDerivatives);
	
	public abstract float validate(ComputeGraph cg, 
			List<Hashtable<String, Matrix>> validationExamples);

}
