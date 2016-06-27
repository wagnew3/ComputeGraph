package test;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;

import graph.ComputeGraph;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Input;
import machineLearning.matrixTransform.MAdd;
import machineLearning.matrixTransform.MMult;
import matrix.FMatrix;
import matrix.Matrix;

public class TestWorking 
{
	
	public static void main(String[] args)
	{
		testNetworkOutput();
	}
	
	static void testNetworkOutput()
	{
		/*
		ComputeGraph cg=new ComputeGraph("standard network");
		cg.addNode("input", new Input());
		cg.addNode("hidden1 weights", new MMult(generateWeightMatrix(2, 2, 2)));
		cg.addNode("hidden1 biases", new MAdd(generateBiasMatrix(2, 1)));
		cg.addNode("hidden1 sigmoid", new Sigmoid());
		cg.addNode("output weights", new MMult(generateWeightMatrix(2, 2, 2)));
		cg.addNode("output biases", new MAdd(generateBiasMatrix(2, 1)));
		cg.addNode("output sigmoid", new Sigmoid());
		
		
		cg.addNode("training outputs", new Input());
		cg.addNode("euclideanCost", new Euclidean());
		
		cg.addEdge("input", "hidden1 weights");
		cg.addEdge("hidden1 weights", "hidden1 biases");
		cg.addEdge("hidden1 biases", "hidden1 sigmoid");
		cg.addEdge("hidden1 sigmoid", "output weights");
		cg.addEdge("output weights", "output biases");
		cg.addEdge("output biases", "output sigmoid");
		cg.addEdge("network output", "output sigmoid", "euclideanCost");
		cg.addEdge("train output", "training outputs", "euclideanCost");
		*/
		
		ComputeGraph cg=new ComputeGraph("standard network");
		cg.addNode("input", new Input());
		cg.addNode("hidden1 weights", new MMult(new FMatrix(new float[][]{new float[]{0.15f, 0.20f}, new float[]{0.25f, 0.30f}})));
		cg.addNode("hidden1 biases", new MAdd(new FMatrix(new float[][]{new float[]{0.35f}, new float[]{0.35f}})));
		cg.addNode("hidden1 sigmoid", new Sigmoid());
		cg.addNode("output weights", new MMult(new FMatrix(new float[][]{new float[]{0.40f, 0.45f}, new float[]{0.50f, 0.55f}})));
		cg.addNode("output biases", new MAdd(new FMatrix(new float[][]{new float[]{0.60f}, new float[]{0.60f}})));
		cg.addNode("output sigmoid", new Sigmoid());
		
		
		cg.addNode("training outputs", new Input());
		cg.addNode("euclideanCost", new Euclidean());
		
		cg.addEdge("input", "hidden1 weights");
		cg.addEdge("hidden1 weights", "hidden1 biases");
		cg.addEdge("hidden1 biases", "hidden1 sigmoid");
		cg.addEdge("hidden1 sigmoid", "output weights");
		cg.addEdge("output weights", "output biases");
		cg.addEdge("output biases", "output sigmoid");
		cg.addEdge("network output", "output sigmoid", "euclideanCost");
		cg.addEdge("train output", "training outputs", "euclideanCost");
		
		List<String> outputVertices=new ArrayList<>();
		outputVertices.add("output sigmoid");
		cg.setOutputVertices(outputVertices);
		
		Hashtable<String, Matrix> inputs=new Hashtable<>();
		inputs.put("input", new FMatrix(new float[][]{new float[]{0.05f}, new float[]{0.10f}}));
		Matrix output=cg.getOutput(inputs).get("output sigmoid");
		
		Hashtable<String, Matrix>[] derivatives=cg.derive(inputs);
		Hashtable<String, Matrix> objectiveDerivatives=derivatives[0];
		Hashtable<String, Matrix> parameterDerivatives=derivatives[1];
		int u=0;
	}
	
	static Matrix generateWeightMatrix(int rows, int cols, int inputSize)
	{
		Matrix weights=new FMatrix(rows, cols);
		NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(inputSize));
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
		NormalDistribution nInvGaussian=new NormalDistribution(0.0, 0.0);
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
