package machineLearning.neuralNetworks;

import java.util.Hashtable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.AbstractGraph;

import graph.ComputeGraph;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Input;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.MAdd;
import machineLearning.matrixTransform.MMult;
import machineLearning.matrixTransform.Split;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class SimpleRecurrentNetwork 
{
	
	ComputeGraph network;
	ComputeGraph unrolledNetwork;
	
	public SimpleRecurrentNetwork(int[] inputShape, int[] outputShape)
	{
		network=new ComputeGraph("SRN network");
		ComputeGraph cg=new ComputeGraph("standard network");
		
		ComputeNode input=cg.addNode("input", new Input());
		ComputeNode memory=cg.addNode("memory", new Input());
		ComputeNode combine=cg.addNode("combine", new Combine(inputShape[0], 0));
		
		ComputeNode weights1=cg.addNode("hidden1 weights", new MMult(generateWeightMatrix(inputShape[0]*inputShape[1], 2*inputShape[0]*inputShape[1], 2*inputShape[0]*inputShape[1])));
		ComputeNode biases1=cg.addNode("hidden1 biases", new MAdd(generateBiasMatrix(inputShape[0]*inputShape[1], 1)));
		ComputeNode sigmoid1=cg.addNode("hidden1 tanH", new TanH());
		
		ComputeNode trainingOutputs=cg.addNode("training outputs", new Input());
		ComputeNode cost=cg.addNode("cost function", new Euclidean());
		
		input.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{combine});
		memory.setInputOutputNode(new ComputeNode[]{memory}, new ComputeNode[]{combine});
		combine.setInputOutputNode(new ComputeNode[]{input, memory}, new ComputeNode[]{weights1});
		weights1.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{biases1});
		biases1.setInputOutputNode(new ComputeNode[]{weights1}, new ComputeNode[]{sigmoid1});
		sigmoid1.setInputOutputNode(new ComputeNode[]{biases1}, new ComputeNode[]{cost});
			
		cost.setInputOutputNode(new ComputeNode[]{sigmoid1, trainingOutputs}, new ComputeNode[]{cost});
		
		trainingOutputs.setInputOutputNode(new ComputeNode[]{trainingOutputs}, new ComputeNode[]{cost});
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
	
	public ComputeGraph unroll(ComputeGraph network, ComputeNode input,
			ComputeNode memoryInput, ComputeNode output, ComputeNode memoryOutput, int unrolls)
	{
		ComputeGraph unrolled=new ComputeGraph("unrolled");
		
	}
	
	protected Hashtable<ComputeGraph, ComputeGraph> cloneNodes(ComputeGraph cg, int cloneNumber)
	{
		Hashtable<ComputeNode, ComputeNode> mappedNodes=new Hashtable<>();
		for(Node node: cg.getEachNode())
		{
			ComputeNode computeNode=(ComputeNode)node;
			ComputeNode clonedNode=cg.addNode(computeNode.getId()+cloneNumber, computeNode.getFunction());
			mappedNodes.put(computeNode, clonedNode);
		}
		
		for(Node node: cg.getEachNode())
		{
			ComputeNode computeNode=(ComputeNode)node;
			ComputeNode[] clonedIn=new ComputeNode[computeNode.inputNodes.length];
			for(int inInd=0; inInd<computeNode.inputNodes.length; inInd++)
			{
				clonedIn[inInd]=mappedNodes.get(computeNode.inputNodes[inInd]);
			}
			ComputeNode[] clonedOut=new ComputeNode[computeNode.outputNodes.length];
			for(int outInd=0; outInd<computeNode.outputNodes.length; outInd++)
			{
				clonedOut[outInd]=mappedNodes.get(computeNode.outputNodes[outInd]);
			}
			mappedNodes.get(computeNode).setInputOutputNode(clonedIn, clonedOut);
		}
		return mappedNodes;
	}

}
