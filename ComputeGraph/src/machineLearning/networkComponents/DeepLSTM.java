package machineLearning.networkComponents;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import machineLearning.activationFunction.Sigmoid;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.*;
import machineLearning.generalFunctions.Constant;
import machineLearning.generalFunctions.Input;
import machineLearning.generalFunctions.Passthrough;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.EbeMult;
import machineLearning.matrixTransform.MAdd;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;
import function.*;

public class DeepLSTM extends RecurrentComputeGraph
{

	public DeepLSTM(String name, int[] inputShape, int[] outputShape, int memorySize) 
	{
		super(name);
		
		ComputeNode inputNode=addNode("input", new Input());
		ComputeNode inputMemoryInitialState=addNode("inputMemoryInitialState", new Constant(new FMatrix(outputShape[0], outputShape[1])));
		ComputeNode inputMemory=addNode("inputMemory", new Passthrough());
		ComputeNode longTermMemoryInitialState=addNode("longTermMemoryInitialState", new Constant(new FMatrix(memorySize, 1)));
		ComputeNode longTermMemoryInput=addNode("longTermMemoryInput", new Passthrough());
		
		ComputeNode combine=addNode("combine", new Combine(inputShape[0], 0));
		
		ComputeNode[] startForgetInterfaces=LayerConstr.addLayers(new int[]{inputShape[0]*inputShape[1]+memorySize, 3*(inputShape[0]*inputShape[1]+memorySize)}, 
				new int[]{3*(inputShape[0]*inputShape[1]+memorySize), memorySize},
				new DifferentiableFunction[]{new Sigmoid(), new Sigmoid()}, this, new String[]{"StartForget0", "StartForget1"});
		ComputeNode startForgetWeights=startForgetInterfaces[0];
		ComputeNode startForgetSigmoid=startForgetInterfaces[1];
		
		ComputeNode startForgetEbeMult=addNode("startForgetEbeMult", new EbeMult());
		
		ComputeNode[] updateAmountInterfaces=LayerConstr.addLayers(new int[]{inputShape[0]*inputShape[1]+memorySize, 3*(inputShape[0]*inputShape[1]+memorySize)}, 
				new int[]{3*(inputShape[0]*inputShape[1]+memorySize), memorySize},
				new DifferentiableFunction[]{new Sigmoid(), new Sigmoid()}, this, new String[]{"UpdateAmount0", "UpdateAmount1"});
		ComputeNode updateAmountWeights=updateAmountInterfaces[0];
		ComputeNode updateAmountSigmoid=updateAmountInterfaces[1];
		
		ComputeNode updateAmountEbeMult=addNode("updateAmountEbeMult", new EbeMult());
		
		ComputeNode[] updateContentInterfaces=LayerConstr.addLayers(new int[]{inputShape[0]*inputShape[1]+memorySize, 3*(inputShape[0]*inputShape[1]+memorySize)}, 
				new int[]{3*(inputShape[0]*inputShape[1]+memorySize), memorySize},
				new DifferentiableFunction[]{new TanH(), new TanH()}, this, new String[]{"UpdateContent0", "UpdateContent1"});
		ComputeNode updateContentWeights=updateContentInterfaces[0];
		ComputeNode updateContentTanH=updateContentInterfaces[1];
		
		ComputeNode updateContentAdd=addNode("updateContentAdd", new MAdd());
		
		ComputeNode[] endForgetInterfaces=LayerConstr.addLayers(new int[]{inputShape[0]*inputShape[1]+memorySize, 3*(inputShape[0]*inputShape[1]+memorySize)}, 
				new int[]{3*(inputShape[0]*inputShape[1]+memorySize), outputShape[0]*outputShape[1]},
				new DifferentiableFunction[]{new Sigmoid(), new Sigmoid()}, this, new String[]{"EndForget0", "EndForget1"});
		ComputeNode endForgetWeights=endForgetInterfaces[0];
		ComputeNode endForgetSigmoid=endForgetInterfaces[1];
		
		ComputeNode endForgetEbeMult=addNode("endForgetEbeMult", new EbeMult());
		
		ComputeNode[] outputInterfaces=LayerConstr.addLayers(new int[]{memorySize}, 
				new int[]{outputShape[0]*outputShape[1]},
				new DifferentiableFunction[]{new TanH()}, this, new String[]{"Output0"});
		ComputeNode outputWeights=outputInterfaces[0];
		ComputeNode outputTanH=outputInterfaces[1];
		
		ComputeNode trainingOutputsNode=addNode("training outputs", new Input());
		ComputeNode cost=addNode("cost function", new CrossEntropy());
		
		inputNode.setInputOutputNode(new ComputeNode[]{inputNode}, new ComputeNode[]{combine});
		inputMemoryInitialState.setInputOutputNode(new ComputeNode[]{}, new ComputeNode[]{inputMemory});
		inputMemory.setInputOutputNode(new ComputeNode[]{inputMemoryInitialState}, new ComputeNode[]{});
		longTermMemoryInitialState.setInputOutputNode(new ComputeNode[]{}, new ComputeNode[]{longTermMemoryInput});
		longTermMemoryInput.setInputOutputNode(new ComputeNode[]{longTermMemoryInitialState}, new ComputeNode[]{startForgetEbeMult, combine});
		combine.setInputOutputNode(new ComputeNode[]{inputNode, longTermMemoryInput}, new ComputeNode[]{startForgetWeights, updateAmountWeights, updateContentWeights, endForgetWeights});
		
		startForgetWeights.setInputNode(new ComputeNode[]{combine});
		startForgetSigmoid.setOutputNode(new ComputeNode[]{startForgetEbeMult});
		startForgetEbeMult.setInputOutputNode(new ComputeNode[]{longTermMemoryInput, startForgetSigmoid}, new ComputeNode[]{updateContentAdd});
		
		updateAmountWeights.setInputNode(new ComputeNode[]{combine});
		updateAmountSigmoid.setOutputNode(new ComputeNode[]{updateAmountEbeMult});
		updateAmountEbeMult.setInputOutputNode(new ComputeNode[]{updateAmountSigmoid, updateContentTanH}, new ComputeNode[]{updateContentAdd});
		
		updateContentWeights.setInputNode(new ComputeNode[]{combine});
		updateContentTanH.setOutputNode(new ComputeNode[]{updateAmountEbeMult});
		
		updateContentAdd.setInputOutputNode(new ComputeNode[]{startForgetEbeMult, updateAmountEbeMult}, new ComputeNode[]{outputWeights});
		
		endForgetWeights.setInputNode(new ComputeNode[]{combine});
		endForgetSigmoid.setOutputNode(new ComputeNode[]{endForgetEbeMult});
		endForgetEbeMult.setInputOutputNode(new ComputeNode[]{outputTanH, endForgetSigmoid}, new ComputeNode[]{cost});
		
		outputWeights.setInputNode(new ComputeNode[]{updateContentAdd});
		outputTanH.setOutputNode(new ComputeNode[]{endForgetEbeMult});
		
		trainingOutputsNode.setInputOutputNode(new ComputeNode[]{trainingOutputsNode}, new ComputeNode[]{cost});
		cost.setInputOutputNode(new ComputeNode[]{endForgetEbeMult, trainingOutputsNode}, new ComputeNode[]{cost});
		
		initialState.add(inputMemoryInitialState);
		initialState.add(longTermMemoryInitialState);
		
		inputNodes.add(inputNode);
		inputNodes.add(trainingOutputsNode);
		
		memoryInNodes.add(longTermMemoryInput);
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
