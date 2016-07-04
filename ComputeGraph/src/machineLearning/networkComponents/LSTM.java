package machineLearning.networkComponents;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import graph.ComputeGraph;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Constant;
import machineLearning.generalFunctions.Input;
import machineLearning.generalFunctions.Passthrough;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.EbeMult;
import machineLearning.matrixTransform.MAdd;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import machineLearning.matrixTransform.Split;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class LSTM extends RecurrentComputeGraph
{

	public LSTM(String name, int[] inputShape, int[] outputShape) 
	{
		super(name);
		
		ComputeNode inputNode=addNode("input", new Input());
		ComputeNode inputMemoryInitialState=addNode("inputMemoryInitialState", new Constant(new FMatrix(outputShape[0], outputShape[1])));
		ComputeNode inputMemory=addNode("inputMemory", new Passthrough());
		ComputeNode longTermMemoryInitialState=addNode("longTermMemoryInitialState", new Constant(new FMatrix(outputShape[0], outputShape[1])));
		ComputeNode longTermMemoryInput=addNode("longTermMemoryInput", new Passthrough());
		
		ComputeNode combine=addNode("combine", new Combine(inputShape[0], 0));
		
		ComputeNode startForgetWeights=addNode("startForgetWeights", new ParamMMult(generateWeightMatrix(outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1])));
		ComputeNode startForgetBiases=addNode("startForgetBiases", new ParamMAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		ComputeNode startForgetSigmoid=addNode("startForgetSigmoid", new Sigmoid());
		ComputeNode startForgetEbeMult=addNode("startForgetEbeMult", new EbeMult());
		
		ComputeNode updateAmountWeights=addNode("updateAmountWeights", new ParamMMult(generateWeightMatrix(outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1])));
		ComputeNode updateAmountBiases=addNode("updateAmountBiases", new ParamMAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		ComputeNode updateAmountSigmoid=addNode("updateAmountSigmoid", new Sigmoid());
		ComputeNode updateAmountEbeMult=addNode("updateAmountEbeMult", new EbeMult());
		
		ComputeNode updateContentWeights=addNode("updateContentWeights", new ParamMMult(generateWeightMatrix(outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1])));
		ComputeNode updateContentBiases=addNode("updateContentBiases", new ParamMAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		ComputeNode updateContentTanH=addNode("updateContentTanH", new TanH());
		ComputeNode updateContentAdd=addNode("updateContentAdd", new MAdd());
		
		ComputeNode endForgetWeights=addNode("endForgetWeights", new ParamMMult(generateWeightMatrix(outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1])));
		ComputeNode endForgetBiases=addNode("endForgetBiases", new ParamMAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		ComputeNode endForgetSigmoid=addNode("endForgetSigmoid", new Sigmoid());
		ComputeNode endForgetEbeMult=addNode("endForgetEbeMult", new EbeMult());
		
		ComputeNode outputWeights=addNode("outputWeights", new ParamMMult(generateWeightMatrix(outputShape[0]*outputShape[1], outputShape[0]*outputShape[1])));
		ComputeNode outputBiases=addNode("outputBiases", new ParamMAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		ComputeNode outputTanH=addNode("outputTanH", new TanH());
		
		ComputeNode trainingOutputsNode=addNode("training outputs", new Input());
		ComputeNode cost=addNode("cost function", new Euclidean());
		
		inputNode.setInputOutputNode(new ComputeNode[]{inputNode}, new ComputeNode[]{combine});
		inputMemoryInitialState.setInputOutputNode(new ComputeNode[]{}, new ComputeNode[]{inputMemory});
		inputMemory.setInputOutputNode(new ComputeNode[]{inputMemoryInitialState}, new ComputeNode[]{combine});
		longTermMemoryInitialState.setInputOutputNode(new ComputeNode[]{}, new ComputeNode[]{longTermMemoryInput});
		longTermMemoryInput.setInputOutputNode(new ComputeNode[]{longTermMemoryInitialState}, new ComputeNode[]{startForgetEbeMult});
		combine.setInputOutputNode(new ComputeNode[]{inputNode, inputMemory}, new ComputeNode[]{startForgetWeights, updateAmountWeights, updateContentWeights, endForgetWeights});
		
		startForgetWeights.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{startForgetBiases});
		startForgetBiases.setInputOutputNode(new ComputeNode[]{startForgetWeights}, new ComputeNode[]{startForgetSigmoid});
		startForgetSigmoid.setInputOutputNode(new ComputeNode[]{startForgetBiases}, new ComputeNode[]{startForgetEbeMult});
		startForgetEbeMult.setInputOutputNode(new ComputeNode[]{longTermMemoryInput, startForgetSigmoid}, new ComputeNode[]{updateContentAdd});
		
		updateAmountWeights.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{updateAmountBiases});
		updateAmountBiases.setInputOutputNode(new ComputeNode[]{updateAmountWeights}, new ComputeNode[]{updateAmountSigmoid});
		updateAmountSigmoid.setInputOutputNode(new ComputeNode[]{updateAmountBiases}, new ComputeNode[]{updateAmountEbeMult});
		updateAmountEbeMult.setInputOutputNode(new ComputeNode[]{updateAmountSigmoid, updateContentTanH}, new ComputeNode[]{updateContentAdd});
		
		updateContentWeights.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{updateContentBiases});
		updateContentBiases.setInputOutputNode(new ComputeNode[]{updateContentWeights}, new ComputeNode[]{updateContentTanH});
		updateContentTanH.setInputOutputNode(new ComputeNode[]{updateContentBiases}, new ComputeNode[]{updateAmountEbeMult});
		
		updateContentAdd.setInputOutputNode(new ComputeNode[]{startForgetEbeMult, updateAmountEbeMult}, new ComputeNode[]{outputWeights});
		
		endForgetWeights.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{endForgetBiases});
		endForgetBiases.setInputOutputNode(new ComputeNode[]{endForgetWeights}, new ComputeNode[]{endForgetSigmoid});
		endForgetSigmoid.setInputOutputNode(new ComputeNode[]{endForgetBiases}, new ComputeNode[]{endForgetEbeMult});
		endForgetEbeMult.setInputOutputNode(new ComputeNode[]{outputTanH, endForgetSigmoid}, new ComputeNode[]{cost});
		
		outputWeights.setInputOutputNode(new ComputeNode[]{updateContentAdd}, new ComputeNode[]{outputBiases});
		outputBiases.setInputOutputNode(new ComputeNode[]{outputWeights}, new ComputeNode[]{outputTanH});
		outputTanH.setInputOutputNode(new ComputeNode[]{outputBiases}, new ComputeNode[]{endForgetEbeMult});
		
		trainingOutputsNode.setInputOutputNode(new ComputeNode[]{trainingOutputsNode}, new ComputeNode[]{cost});
		cost.setInputOutputNode(new ComputeNode[]{endForgetEbeMult, trainingOutputsNode}, new ComputeNode[]{cost});
		
		initialState.add(inputMemoryInitialState);
		initialState.add(longTermMemoryInitialState);
		
		inputNodes.add(inputNode);
		inputNodes.add(trainingOutputsNode);
		
		memoryInNodes.add(inputMemory);
		memoryInNodes.add(longTermMemoryInput);
		
		memoryOutNodes.add(endForgetEbeMult);
		memoryOutNodes.add(updateContentAdd);
		
		computeOuts.add(endForgetEbeMult);
		
		objectives.add(cost);
		
		trainingOutputNodes.add(trainingOutputsNode);
	}
	
	static Matrix generateWeightMatrix(int rows, int cols)
	{
		int inputSize=cols;
		Matrix weights=new FMatrix(rows, cols);
		RandomGenerator random=new JDKRandomGenerator();
		random.setSeed(521);
		NormalDistribution nInvGaussian=new NormalDistribution(random, 0.0, 1.0/Math.sqrt(inputSize));
		for(int rowIndex=0; rowIndex<weights.getRows(); rowIndex++)
		{
			for(int colIndex=0; colIndex<weights.getCols(); colIndex++)
			{
				weights.set(rowIndex, colIndex, (float)nInvGaussian.sample());
			}
		}
		return weights;
	}
	
	static Matrix generateBiasMatrix(int rows, int cols)
	{
		Matrix weights=new FMatrix(rows, cols);
		RandomGenerator random=new JDKRandomGenerator();
		random.setSeed(521);
		NormalDistribution nInvGaussian=new NormalDistribution(random, 0.0, 1.0);
		for(int rowIndex=0; rowIndex<weights.getRows(); rowIndex++)
		{
			for(int colIndex=0; colIndex<weights.getCols(); colIndex++)
			{
				weights.set(rowIndex, colIndex, (float)nInvGaussian.sample());
			}
		}
		return weights;
	}

}
