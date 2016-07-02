package machineLearning.neuralNetworks;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.AbstractGraph;

import function.Function;
import graph.ComputeGraph;
import graph.InputSizeComputeGraph;
import machineLearning.Learner.ExampleBatchDerivativeOptimizer;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Constant;
import machineLearning.generalFunctions.Input;
import machineLearning.generalFunctions.Passthrough;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.MAdd;
import machineLearning.matrixTransform.MMult;
import machineLearning.matrixTransform.Split;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class SimpleRecurrentNetwork 
{
	
	public ComputeGraph network;
	public ComputeGraph unrolledNetwork;
	List<ComputeNode> unrolledInputs;
	public Constant initialState;
	ComputeNode inputNode;
	ComputeNode memoryNode;
	ComputeNode sigmoid1Node;
	ComputeNode[] unrolledObjectiveNodes;
	boolean trainOutputMode;
	int unrolls;
	
	public SimpleRecurrentNetwork(int[] inputShape, int[] outputShape, int unrolls)
	{
		this.unrolls=unrolls;
		initialState=new Constant(new FMatrix(outputShape[0], outputShape[1]));
		network=new ComputeGraph("SRN network");
		trainOutputMode=true;

		inputNode=network.addNode("input", new Input());
		memoryNode=network.addNode("memory passthrough", new Passthrough());
		ComputeNode combine=network.addNode("combine", new Combine(inputShape[0], 0));
		
		ComputeNode weights1=network.addNode("hidden1 weights", new MMult(generateWeightMatrix(outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1], inputShape[0]*inputShape[1]+outputShape[0]*outputShape[1])));
		ComputeNode biases1=network.addNode("hidden1 biases", new MAdd(generateBiasMatrix(outputShape[0]*outputShape[1], 1)));
		sigmoid1Node=network.addNode("hidden1 tanH", new TanH());
		
		ComputeNode trainingOutputs=network.addNode("training outputs", new Input());
		ComputeNode cost=network.addNode("cost function", new Euclidean());
		
		inputNode.setInputOutputNode(new ComputeNode[]{inputNode}, new ComputeNode[]{combine});
		memoryNode.setOutputNode(new ComputeNode[]{combine});
		combine.setInputOutputNode(new ComputeNode[]{inputNode, memoryNode}, new ComputeNode[]{weights1});
		weights1.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{biases1});
		biases1.setInputOutputNode(new ComputeNode[]{weights1}, new ComputeNode[]{sigmoid1Node});
		sigmoid1Node.setInputNode(new ComputeNode[]{biases1});
		sigmoid1Node.setOutputNode(new ComputeNode[]{cost});
		trainingOutputs.setInputNode(new ComputeNode[]{trainingOutputs});
		trainingOutputs.setOutputNode(new ComputeNode[]{cost});
		cost.setInputOutputNode(new ComputeNode[]{sigmoid1Node, trainingOutputs}, new ComputeNode[]{cost});
		
		Object[] unrolledData=unroll(network, inputNode, memoryNode, sigmoid1Node, 
				cost, trainingOutputs, unrolls);
		unrolledNetwork=((ComputeGraph)(unrolledData[0]));
		unrolledInputs=(List<ComputeNode>)(unrolledData[1]);
		unrolledObjectiveNodes=(ComputeNode[])(unrolledData[2]);
		
		List<ComputeNode> outputVertices=new ArrayList<>();
		outputVertices.add(cost);
		network.setOutputVertices(outputVertices);
	}
	
		   //[0]=output, [1]=newRememberedState
	public Matrix[] getOutput(Matrix input, Matrix rememberedState)
	{
		if(trainOutputMode)
		{
			List<ComputeNode> outputVertices=new ArrayList<>();
			outputVertices.add(sigmoid1Node);
			network.setOutputVertices(outputVertices);
			trainOutputMode=false;
		}
		Hashtable<ComputeNode, Matrix> inputs=new Hashtable<>();
		inputs.put(inputNode, input);
		inputs.put(memoryNode, rememberedState);
		Hashtable<ComputeNode, Matrix> outputs=network.getOutput(inputs);
		return new Matrix[]{outputs.get(inputNode), outputs.get(memoryNode)};
	}
	
	public void train(ExampleBatchDerivativeOptimizer trainer, Matrix[][][] trainingInputs,
			Matrix[][][] trainingOutputs, Matrix[][][] validationInputs, Matrix[][][] validationOutputs)
	{
		List<Hashtable<ComputeNode, Matrix>> trainingExamples=new ArrayList<>();
		for(int runInd=0; runInd<trainingInputs.length; runInd++)
		{
			for(int runTimeInd=0; runTimeInd<trainingInputs[runInd].length; runTimeInd++)
			{
				Hashtable<ComputeNode, Matrix> inputsTable=new Hashtable<>();
				for(int runTimeTrainingInd=runTimeInd; 
						runTimeTrainingInd>(int)Math.max(-1, runTimeInd-unrolls); 
						runTimeTrainingInd--)
				{
					inputsTable.put(unrolledInputs.get(2*(runTimeInd-runTimeTrainingInd)), 
							trainingInputs[runInd][runTimeTrainingInd][0]);
					inputsTable.put(unrolledInputs.get(1+2*(runTimeInd-runTimeTrainingInd)), 
							trainingOutputs[runInd][runTimeTrainingInd][0]);
				}
				trainingExamples.add(inputsTable);
			}
		}
		
		List<Hashtable<ComputeNode, Matrix>> validationExamples=new ArrayList<>();
		for(int runInd=0; runInd<validationInputs.length; runInd++)
		{
			for(int runTimeInd=0; runTimeInd<validationInputs[runInd].length; runTimeInd++)
			{
				Hashtable<ComputeNode, Matrix> inputsTable=new Hashtable<>();
				for(int runTimeTrainingInd=runTimeInd; 
						runTimeTrainingInd>(int)Math.max(-1, runTimeInd-unrolls); 
						runTimeTrainingInd--)
				{
					inputsTable.put(unrolledInputs.get(2*(runTimeInd-runTimeTrainingInd)), 
							validationInputs[runInd][runTimeTrainingInd][0]);
					inputsTable.put(unrolledInputs.get(1+2*(runTimeInd-runTimeTrainingInd)), 
							validationOutputs[runInd][runTimeTrainingInd][0]);
				}
				validationExamples.add(inputsTable);
			}
		}
		
		trainer.setExamples(trainingExamples, validationExamples);
		trainer.optimize(unrolledNetwork, unrolledObjectiveNodes);
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
	
			//ComputeGraph, List<ComputeNode> inputs, ComputeNode objective
	public Object[] unroll(ComputeGraph network, ComputeNode input,
			ComputeNode memoryInput, ComputeNode memoryOutput, 
			ComputeNode output, ComputeNode trainingInput, int unrolls)
	{
		ComputeGraph unrolled=new ComputeGraph("unrolled");
		ComputeNode[] objectiveInputs=new ComputeNode[2*unrolls];
		
		Hashtable<ComputeNode, ComputeNode>[] clonedNodesMap=new Hashtable[unrolls];
		for(int unrollCount=0; unrollCount<unrolls; unrollCount++)
		{
			clonedNodesMap[unrollCount]=cloneNodes(network, unrolled, unrollCount);
		}
		
		ComputeNode[] objectiveNodes=new ComputeNode[unrolls];
		List<ComputeNode> outputNodes=new ArrayList<>();
		List<ComputeNode> inputVertices=new ArrayList<>();
		for(int unrollCount=0; unrollCount<unrolls; unrollCount++)
		{
			if(unrollCount==0)
			{
				ComputeNode initialStateNode=unrolled.addNode("initial state", initialState);
				initialStateNode.setInputOutputNode(new ComputeNode[]{}, new ComputeNode[]{clonedNodesMap[0].get(memoryInput)});
				clonedNodesMap[0].get(memoryInput).setInputNode(new ComputeNode[]{initialStateNode});
			}
			else
			{
				clonedNodesMap[unrollCount-1].get(memoryOutput).addOutputNode(new ComputeNode[]{clonedNodesMap[unrollCount].get(memoryInput)});
				clonedNodesMap[unrollCount].get(memoryInput).setInputNode(new ComputeNode[]{clonedNodesMap[unrollCount-1].get(memoryOutput)});
			}
			inputVertices.add(clonedNodesMap[unrollCount].get(input));
			inputVertices.add(clonedNodesMap[unrollCount].get(trainingInput));
			objectiveInputs[2*unrollCount]=clonedNodesMap[unrollCount].get(output);
			objectiveInputs[2*unrollCount+1]=clonedNodesMap[unrollCount].get(trainingInput);
			
			outputNodes.add(clonedNodesMap[unrollCount].get(output));
			objectiveNodes[unrollCount]=clonedNodesMap[unrollCount].get(output);
		}
		unrolled.setOutputVertices(outputNodes);
		
		return new Object[]{unrolled, inputVertices, objectiveNodes};
	}
	
	protected Hashtable<ComputeNode, ComputeNode> cloneNodes(ComputeGraph cloneGraph, ComputeGraph addGraph, int cloneNumber)
	{
		Hashtable<ComputeNode, ComputeNode> mappedNodes=new Hashtable<>();
		for(Node node: cloneGraph.getEachNode())
		{
			ComputeNode computeNode=(ComputeNode)node;
			ComputeNode clonedNode=addGraph.addNode(computeNode.getId()+cloneNumber, computeNode.getFunction());
			mappedNodes.put(computeNode, clonedNode);
		}
		
		for(Node node: cloneGraph.getEachNode())
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
