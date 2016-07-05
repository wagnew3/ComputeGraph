package machineLearning.Learner;

import java.util.Hashtable;
import java.util.List;

import function.UpdatableDifferentiableFunction;
import graph.ComputeGraph;
import machineLearning.generalFunctions.Constant;
import matrix.FMatrix;
import matrix.Matrix;
import test.TestRecurrentNeuralNetworks;
import vertex.ComputeNode;

public class Nestrov extends ExampleBatchDerivativeOptimizer
{
	
	protected float learningRate;
	protected float momentumParam;
	
	Hashtable<ComputeNode, Matrix> momentums;
	
	public Nestrov(List<Hashtable<ComputeNode, Matrix>> examples, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples,
			int batchSize, int numberEpochs, float learningRate,
			float momentumParam) 
	{
		super(examples, validationExamples, batchSize, numberEpochs);
		this.learningRate=learningRate;
		this.momentumParam=momentumParam;
		momentums=new Hashtable<>();
	}

	@Override
	public void updateParameters(ComputeGraph cg, Hashtable<ComputeNode, Matrix> batchDerivatives)
	{
		float batchSizeScale=1.0f/getBatchSize();
		for(ComputeNode vertex: batchDerivatives.keySet())
		{
			Matrix gradient=batchDerivatives.get(vertex);
			if(momentums.get(vertex)==null)
			{
				momentums.put(vertex, new FMatrix(gradient.getRows(), gradient.getCols()));
			}
			Matrix momentum=momentums.get(vertex);
			momentum=momentum.omscal(momentumParam).omad(gradient.omscal(learningRate*batchSizeScale));

			UpdatableDifferentiableFunction updateFunction
				=((UpdatableDifferentiableFunction)(vertex).getFunction());
			Matrix update=updateFunction.getParameter().omsub(momentum);
			updateFunction.updateParameter(update);
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
			Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> output=(Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>>)cg.compute(validationExample)[0];
			for(int objectiveInd=0; objectiveInd<objectives.length; objectiveInd++)
			{
				if(output.get(objectives[objectiveInd])!=null)
				{
					totalError+=output.get(objectives[objectiveInd]).get(objectives[objectiveInd]).get(0, 0);
					numberErrors++;
					
					/*
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
					*/
					
					Matrix netOut=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]);
					Matrix trainOut=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[1]);
					int highestNetInd=-1;
					int highestTrainInd=-1;
					float highestNet=Float.NEGATIVE_INFINITY;
					float highestTrain=Float.NEGATIVE_INFINITY;
					
					for(int outInd=0; outInd<netOut.getLen(); outInd++)
					{
						if(highestNet<netOut.get(outInd, 0))
						{
							highestNet=netOut.get(outInd, 0);
							highestNetInd=outInd;
						}
						if(highestTrain<trainOut.get(outInd, 0))
						{
							highestTrain=trainOut.get(outInd, 0);
							highestTrainInd=outInd;
						}
					}
					
					if(highestNetInd==highestTrainInd)
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
			
			int length=25;
			byte[] testInput="Here comes".getBytes();
			String testOutput="[Here comes]";
			Matrix shortMemory=((Constant)((ComputeNode)TestRecurrentNeuralNetworks.recurrentNetwork.getNode("inputMemoryInitialState")).getFunction()).getParameter();
			Matrix longMemory=((Constant)((ComputeNode)TestRecurrentNeuralNetworks.recurrentNetwork.getNode("longTermMemoryInitialState")).getFunction()).getParameter();
			for(int ind=0; ind<25; ind++)
			{
				if(ind<testInput.length)
				{
					Matrix inputMat=TestRecurrentNeuralNetworks.letterToVec(testInput[ind]);
					Matrix[][] outputs
						=TestRecurrentNeuralNetworks.recurrentNetwork
							.getOutput(new Matrix[]{inputMat}, new Matrix[]{shortMemory, longMemory});
					Matrix outputMat=outputs[0][0];
					shortMemory=outputs[1][0];
					longMemory=outputs[1][1];
				}
				else
				{
					Matrix inputMat=TestRecurrentNeuralNetworks.letterToVec((byte)testOutput.charAt(testOutput.length()-1));
					Matrix[][] outputs
						=TestRecurrentNeuralNetworks.recurrentNetwork
							.getOutput(new Matrix[]{inputMat}, new Matrix[]{shortMemory, longMemory});
					Matrix outputMat=outputs[0][0];
					shortMemory=outputs[1][0];
					longMemory=outputs[1][1];
					testOutput+=TestRecurrentNeuralNetworks.vecToLetter(outputMat);
				}
			}
			System.out.println("Recurrent Neural net text: "+testOutput);
			
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
