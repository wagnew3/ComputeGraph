package machineLearning.neuralNetworks;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;

import graph.ComputeGraph;
import machineLearning.Learner.ExampleBatchDerivativeOptimizer;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Constant;
import machineLearning.generalFunctions.Input;
import machineLearning.generalFunctions.Passthrough;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import machineLearning.matrixTransform.Split;
import machineLearning.networkComponents.RecurrentComputeGraph;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class RecurrentNetwork extends ComputeGraph
{
	
	public ComputeGraph unrolledNetwork;
	List<ComputeNode> unrolledInputs;
	public List<Constant> initialStates;
	List<ComputeNode> inputNodes;
	List<ComputeNode> memoryNodes;
	List<ComputeNode> computeOuts;
	ComputeNode[] unrolledObjectiveNodes;
	boolean trainOutputMode;
	int unrolls;
	
	public RecurrentNetwork(String name, ComputeGraph reccurentUnit, List<ComputeNode> initialState,
			List<ComputeNode> inputNodes, List<ComputeNode> memoryInNodes, List<ComputeNode> memoryOutNodes, 
			List<ComputeNode> computeOuts, List<ComputeNode> objectives, List<ComputeNode> trainingOutputNodes, int unrolls)
	{
		super(name);
		init(reccurentUnit, initialState,
				inputNodes, memoryInNodes, memoryOutNodes, 
				computeOuts, objectives, trainingOutputNodes, unrolls);
	}
	
	public RecurrentNetwork(String name, RecurrentComputeGraph rcg, int unrolls)
	{
		super(name);
		init(rcg, rcg.initialState,
				rcg.inputNodes, rcg.memoryInNodes, rcg.memoryOutNodes, 
				rcg.computeOuts, rcg.objectives, rcg.trainingOutputNodes, unrolls);
	}
	
	protected void init(ComputeGraph reccurentUnit, List<ComputeNode> initialState,
			List<ComputeNode> inputNodes, List<ComputeNode> memoryInNodes, List<ComputeNode> memoryOutNodes, 
			List<ComputeNode> computeOuts, List<ComputeNode> objectives, List<ComputeNode> trainingOutputNodes, 
			int unrolls)
	{
		this.unrolls=unrolls;
		trainOutputMode=true;

		cloneNodes(reccurentUnit, this, "");
		
		Object[] unrolledData=unroll(reccurentUnit, initialState, inputNodes, memoryInNodes, memoryOutNodes, 
				objectives, trainingOutputNodes, unrolls);
		unrolledNetwork=((ComputeGraph)(unrolledData[0]));
		unrolledInputs=(List<ComputeNode>)(unrolledData[1]);
		unrolledObjectiveNodes=(ComputeNode[])(unrolledData[2]);
		
		setOutputVertices(computeOuts);
	}
	
		   //[0]=output, [1]=newRememberedState
	public Hashtable<ComputeNode, Matrix> getOutput(Hashtable<ComputeNode, Matrix> inputs, Hashtable<ComputeNode, Matrix> rememberedState)
	{
		if(trainOutputMode)
		{
			setOutputVertices(computeOuts);
			trainOutputMode=false;
		}
		Hashtable<ComputeNode, Matrix> inputs=new Hashtable<>();
		inputs.put(inputNode, input);
		inputs.put(memoryNode, rememberedState);
		Hashtable<ComputeNode, Matrix> outputs=getOutput(inputs);
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
					inputsTable.put(unrolledInputs.get(2*(runTimeTrainingInd)), 
							trainingInputs[runInd][runTimeTrainingInd][0]);
					inputsTable.put(unrolledInputs.get(1+2*(runTimeTrainingInd)), 
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
					inputsTable.put(unrolledInputs.get(2*(runTimeTrainingInd)), 
							validationInputs[runInd][runTimeTrainingInd][0]);
					inputsTable.put(unrolledInputs.get(1+2*(runTimeTrainingInd)), 
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
	public Object[] unroll(ComputeGraph network, List<ComputeNode> initialStates,
			List<ComputeNode> inputs,
			List<ComputeNode> memoryInputs, List<ComputeNode> memoryOutputs, 
			List<ComputeNode> outputs, List<ComputeNode> trainingInputs, int unrolls)
	{
		ComputeGraph unrolled=new ComputeGraph("unrolled");
		

		Hashtable<ComputeNode, ComputeNode>[] clonedNodesMap=new Hashtable[unrolls];
		for(int unrollCount=0; unrollCount<unrolls; unrollCount++)
		{
			clonedNodesMap[unrollCount]=cloneNodes(network, unrolled, " "+unrollCount);
			if(unrollCount==0)
			{
				for(ComputeNode intitialState: initialStates)
				{
					network.removeNode(intitialState);
				}
			}
		}
		
		ComputeNode[] objectiveInputs=new ComputeNode[2*unrolls];
		ComputeNode[] objectiveNodes=new ComputeNode[unrolls*outputs.size()];
		List<ComputeNode> outputNodes=new ArrayList<>();
		List<ComputeNode> inputVertices=new ArrayList<>();
		int unrollAddedInd=0;
		for(int unrollCount=0; unrollCount<unrolls; unrollCount++)
		{
			if(unrollCount>0)
			{
				for(int memoryInd=0; memoryInd<memoryOutputs.size(); memoryInd++)
				{
					clonedNodesMap[unrollCount-1].get(memoryOutputs.get(memoryInd)).addOutputNode(new ComputeNode[]{clonedNodesMap[unrollCount].get(memoryInputs.get(memoryInd))});
					clonedNodesMap[unrollCount].get(memoryInputs.get(memoryInd)).setInputNode(new ComputeNode[]{clonedNodesMap[unrollCount-1].get(memoryOutputs.get(memoryInd))});
				}
			}
			for(ComputeNode inputNode: inputs)
			{
				inputVertices.add(clonedNodesMap[unrollCount].get(inputNode));
			}
			for(ComputeNode trainingInputNode: trainingInputs)
			{
				//inputVertices.add(clonedNodesMap[unrollCount].get(trainingInputNode));
			}
			
			
			for(int outInd=0; outInd<outputs.size(); outInd++)
			{
				outputNodes.add(clonedNodesMap[unrollCount].get(outputs.get(outInd)));
				objectiveNodes[unrollAddedInd]=clonedNodesMap[unrollCount].get(outputs.get(outInd));
				objectiveInputs[2*unrollAddedInd]=clonedNodesMap[unrollCount].get(outputs.get(outInd));
				objectiveInputs[2*unrollAddedInd+1]=clonedNodesMap[unrollCount].get(trainingInputs.get(outInd));
				unrollAddedInd++;
			}
		}
		unrolled.setOutputVertices(outputNodes);
		
		return new Object[]{unrolled, inputVertices, objectiveNodes};
	}
	
	protected Hashtable<ComputeNode, ComputeNode> cloneNodes(ComputeGraph cloneGraph,
			ComputeGraph addGraph, String cloneIDAdd)
	{
		Hashtable<ComputeNode, ComputeNode> mappedNodes=new Hashtable<>();
		for(Node node: cloneGraph.getEachNode())
		{
			ComputeNode computeNode=(ComputeNode)node;
			ComputeNode clonedNode=addGraph.addNode(computeNode.getId()+cloneIDAdd, computeNode.getFunction());
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
