package machineLearning.neuralNetworks;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import graph.ComputeGraph;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Input;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import matrix.FMatrix;
import matrix.Matrix;

public class LSTM 
{
	
	ComputeGraph network;
	
	public LSTM(int[] inputShape, int[] outputShape)
	{
		network=new ComputeGraph("LSTM network");
		network.addNode("input", new Input());
		network.addNode("hidden1 weights", new ParamMMult(generateWeightMatrix(10, 28*28, 28*28)));
		network.addNode("hidden1 biases", new ParamMAdd(generateBiasMatrix(10, 1)));
		network.addNode("hidden1 sigmoid", new TanH());
		network.addNode("output weights", new ParamMMult(generateWeightMatrix(10, 10, 10)));
		network.addNode("output biases", new ParamMAdd(generateBiasMatrix(10, 1)));
		network.addNode("output sigmoid", new TanH());
		
		
		network.addNode("training outputs", new Input());
		network.addNode("cost function", new Euclidean());
		
		network.addEdge("input", "hidden1 weights");
		network.addEdge("hidden1 weights", "hidden1 biases");
		network.addEdge("hidden1 biases", "hidden1 sigmoid");
		network.addEdge("hidden1 sigmoid", "output weights");
		network.addEdge("output weights", "output biases");
		network.addEdge("output biases", "output sigmoid");
		network.addEdge("network output", "output sigmoid", "cost function");
		network.addEdge("train output", "training outputs", "cost function");
	}
	
	static Matrix generateWeightMatrix(int rows, int cols, int inputSize)
	{
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
