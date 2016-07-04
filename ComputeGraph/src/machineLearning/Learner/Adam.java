package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import function.UpdatableDifferentiableFunction;
import graph.ComputeGraph;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class Adam extends ExampleBatchDerivativeOptimizer
{
	
	private static float a=0.001f;
	private static float B1=0.9f;
	private static float B2=0.999f;
	private static float e=0.00000001f;
	
	private Hashtable<ComputeNode, Matrix> ms;
	private Hashtable<ComputeNode, Matrix> vs;
	private Hashtable<ComputeNode, Matrix> thetas;
	private double stepNumber=0;

	public Adam(List<Hashtable<ComputeNode, Matrix>> examples, List<Hashtable<ComputeNode, Matrix>> validationExamples,
			int batchSize, int numberEpochs) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		ms=new Hashtable<ComputeNode, Matrix>();
		vs=new Hashtable<ComputeNode, Matrix>();
		thetas=new Hashtable<ComputeNode, Matrix>();
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<ComputeNode, Matrix> batchDerivatives) 
	{
		stepNumber++;
		for(ComputeNode node: batchDerivatives.keySet())
		{
			Matrix curDers=batchDerivatives.get(node);
			if(ms.get(node)==null)
			{
				ms.put(node, new FMatrix(curDers.getRows(), curDers.getCols()));
			}
			Matrix m=ms.get(node);
			if(vs.get(node)==null)
			{
				vs.put(node, new FMatrix(curDers.getRows(), curDers.getCols()));
			}
			Matrix v=vs.get(node);
			if(thetas.get(node)==null)
			{
				thetas.put(node, new FMatrix(curDers.getRows(), curDers.getCols()));
			}
			Matrix theta=thetas.get(node);
			
			m=m.omscal(B1).omad(curDers.scal((1.0f-B1), 1, new FMatrix(curDers.getRows(), curDers.getCols())));
			v=v.omscal(B2).omad(curDers.ebemult(curDers, new FMatrix(curDers.getRows(), curDers.getCols()))).omscal((1.0f-B2));
			Matrix mhat=m.scal(1.0f/(1.0f-(float)Math.pow(B1, stepNumber)), 1, new FMatrix(curDers.getRows(), curDers.getCols()));
			Matrix vhat=v.scal(1.0f/(1.0f-(float)Math.pow(B2, stepNumber)), 1, new FMatrix(curDers.getRows(), curDers.getCols()));
			vhat.oebePow(0.5f).osADD(e);
			Matrix update=mhat.oebeDiv(vhat).omscal(a);
			
			for(int updateRow=0; updateRow<update.getRows(); updateRow++)
			{
				if(!Float.isFinite(update.get(updateRow, 0)))
				{
					int u=0;
				}
			}
					
			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)node.getFunction());
			updateFunction.updateParameter(updateFunction.getParameter().omsub(update));
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
		for(int validationExampleInd=0; validationExampleInd<validationExamples.size(); validationExampleInd++)
		{
			Hashtable<ComputeNode, Matrix> validationExample=validationExamples.get(validationExampleInd);
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
