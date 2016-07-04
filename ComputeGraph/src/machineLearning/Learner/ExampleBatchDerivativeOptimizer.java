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
	
	static int numberThreads=2;
	
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
	
	public void setExamples(List<Hashtable<ComputeNode, Matrix>> examples, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples)
	{
		this.examples=examples;
		this.validationExamples=validationExamples;
	}

	@Override
	public void optimize(ComputeGraph cg, ComputeNode[] objectives) 
	{
		List<Hashtable<ComputeNode, Matrix>> randomizedExamples=new ArrayList<>();
		randomizedExamples.addAll(examples);
		
		DeriveThread[] threads=new DeriveThread[numberThreads];
		for(int threadInd=0; threadInd<threads.length; threadInd++)
		{
			threads[threadInd]=new DeriveThread(cg);
		}
		
		for(int epoch=0; epoch<numberEpochs; epoch++)
		{
			Collections.shuffle(randomizedExamples);
			//System.out.println("unrandomized inputs");
						
			for(int sampleInd=0; sampleInd<randomizedExamples.size(); sampleInd+=batchSize)
			{
				Hashtable<ComputeNode, Matrix> batchParameterDerivatives=new Hashtable<>();
				Hashtable<ComputeNode, Matrix>[] batchExamples=new Hashtable[Math.min(batchSize, randomizedExamples.size()-sampleInd)];
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					batchExamples[batchInd]=randomizedExamples.get(batchInd+sampleInd);
				}
				
				for(int threadInd=0; threadInd<threads.length; threadInd++)
				{
					threads[threadInd].reset();;
				}
				for(int threadInd=0; threadInd<threads.length; threadInd++)
				{
					threads[threadInd].setExamples(batchExamples);
				}
				while(!DeriveThread.examplesEmpty())
				{
					try 
					{
						Thread.sleep(1);
					} 
					catch (InterruptedException e) 
					{
						e.printStackTrace();
					}
				}
				
				for(int threadInd=0; threadInd<threads.length; threadInd++)
				{
					for(ComputeNode nodeDeriv: threads[threadInd].batchParameterDerivatives.keySet())
					{
						if(batchParameterDerivatives.get(nodeDeriv)==null)
						{
							batchParameterDerivatives.put(nodeDeriv, threads[threadInd].batchParameterDerivatives.get(nodeDeriv));
						}
						else
						{
							batchParameterDerivatives.get(nodeDeriv).omad(threads[threadInd].batchParameterDerivatives.get(nodeDeriv));
						}
					}
				}
				
				
				/*
				for(int batchInd=0; batchInd<Math.min(batchSize, randomizedExamples.size()-sampleInd); batchInd++)
				{
					Hashtable<ComputeNode, Matrix> parameterDerivatives=cg.derive(batchExamples[batchInd]);
					for(ComputeNode nodeDeriv: parameterDerivatives.keySet())
					{
						if(batchParameterDerivatives.get(nodeDeriv)==null)
						{
							batchParameterDerivatives.put(nodeDeriv, parameterDerivatives.get(nodeDeriv));
						}
						else
						{
							batchParameterDerivatives.get(nodeDeriv).omad(parameterDerivatives.get(nodeDeriv));
						}
					}
				}
				*/
				updateParameters(cg, batchParameterDerivatives);
			}
			float validationError=validate(cg, validationExamples, objectives);
			System.out.println("Epoch: "+epoch+" Average Validation Error: "+validationError);
		}
		for(int threadInd=0; threadInd<threads.length; threadInd++)
		{
			threads[threadInd].stop=true;
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
			List<Hashtable<ComputeNode, Matrix>> validationExamples, ComputeNode[] objectives);

}

class DeriveThread extends Thread
{
	
	private static volatile Hashtable<ComputeNode, Matrix>[] batchExamples;
	private static volatile int exampleInd;
	public volatile boolean stop;
	private volatile ComputeGraph computeGraph;
	public volatile Hashtable<ComputeNode, Matrix> batchParameterDerivatives;
	
	
	public DeriveThread(ComputeGraph computeGraph)
	{
		this.batchExamples=new Hashtable[0];
		this.computeGraph=computeGraph;
		stop=false;
		start();
	}
	
	@Override
	public void run()
	{
		while(!stop)
		{
			Hashtable<ComputeNode, Matrix> example=getNextExample();
			if(example!=null)
			{
				Hashtable<ComputeNode, Matrix> parameterDerivatives=computeGraph.derive(example);
				for(ComputeNode nodeDeriv: parameterDerivatives.keySet())
				{
					synchronized(batchParameterDerivatives)
					{
						if(batchParameterDerivatives.get(nodeDeriv)==null)
						{
							batchParameterDerivatives.put(nodeDeriv, parameterDerivatives.get(nodeDeriv));
						}
						else
						{
							batchParameterDerivatives.get(nodeDeriv).omad(parameterDerivatives.get(nodeDeriv));
						}
					}
				}
			}
		}
	}
	
	public static Hashtable<ComputeNode, Matrix> getNextExample()
	{
		synchronized(batchExamples)
		{
			if(exampleInd<batchExamples.length)
			{
				Hashtable<ComputeNode, Matrix> example=batchExamples[exampleInd];
				batchExamples[exampleInd]=null;
				if(exampleInd<batchExamples.length-1)
				{
					exampleInd++;
				}
				return example;
			}
			else
			{
				return null;
			}
		}
	}
	
	public static boolean examplesEmpty()
	{
		synchronized(batchExamples)
		{
			return batchExamples[batchExamples.length-1]==null;
		}
	}
	
	public void reset()
	{
		exampleInd=0;
		synchronized(batchParameterDerivatives)
		{
			batchParameterDerivatives=new Hashtable<>();
		}
	}
	
	public void setExamples(Hashtable<ComputeNode, Matrix>[] batchExamples)
	{
		synchronized(batchExamples)
		{
			this.batchExamples=batchExamples;
		}
	}
	
}
