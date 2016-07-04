package machineLearning.networkComponents;

import graph.*;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import matrix.FMatrix;
import matrix.Matrix;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import function.*;
import vertex.*;

public class LayerConstr 
{
	
	public static ComputeNode[] addLayers(int[] inputShapes, int[] outputShapes, 
			DifferentiableFunction[] activationFunctions, ComputeGraph toAddTo, String[] layerNames)
	{
		ComputeNode[] weights=new ComputeNode[inputShapes.length];
		ComputeNode[] activations=new ComputeNode[inputShapes.length];
		
		for(int layerInd=0; layerInd<inputShapes.length; layerInd++)
		{
			ComputeNode[] layerInterfaces=addLayer(inputShapes[layerInd], outputShapes[layerInd], 
					activationFunctions[layerInd], toAddTo, layerNames[layerInd]);
			weights[layerInd]=layerInterfaces[0];
			activations[layerInd]=layerInterfaces[1];
		}
		
		for(int layerInd=0; layerInd<inputShapes.length; layerInd++)
		{
			if(layerInd<inputShapes.length-1)
			{
				activations[layerInd].setOutputNode(new ComputeNode[]{weights[layerInd+1]});
			}
			if(layerInd>0)
			{
				weights[layerInd].setInputNode(new ComputeNode[]{activations[layerInd-1]});
			}
		}
		
		return new ComputeNode[]{weights[0], activations[activations.length-1]};
	}
	
	public static ComputeNode[] addLayer(int inputShape, int outputShape, 
			DifferentiableFunction activationFunction, ComputeGraph toAddTo, String layerName)
	{
		ComputeNode weights=toAddTo.addNode(layerName+" Weights", new ParamMMult(generateWeightMatrix(outputShape, inputShape)));
		ComputeNode biases=toAddTo.addNode(layerName+" Biases", new ParamMAdd(generateBiasMatrix(outputShape, 1)));
		ComputeNode activation=toAddTo.addNode(layerName+" Sigmoid", new Sigmoid());
		
		weights.setOutputNode(new ComputeNode[]{biases});
		biases.setInputOutputNode(new ComputeNode[]{weights}, new ComputeNode[]{activation});
		activation.setInputNode(new ComputeNode[]{biases});
		
		return new ComputeNode[]{weights, activation};
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
