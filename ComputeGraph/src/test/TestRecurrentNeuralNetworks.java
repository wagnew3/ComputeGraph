package test;

import java.awt.BorderLayout;
import java.awt.Component;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;

import javax.swing.JFrame;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.graphstream.ui.swingViewer.View;
import org.graphstream.ui.swingViewer.Viewer;

import graph.ComputeGraph;
import machineLearning.Learner.*;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Constant;
import machineLearning.generalFunctions.Input;
import machineLearning.generalFunctions.Passthrough;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import machineLearning.matrixTransform.Split;
import machineLearning.networkComponents.DeepLSTM;
import machineLearning.networkComponents.LSTM;
import machineLearning.networkComponents.RecurrentComputeGraph;
import machineLearning.neuralNetworks.RecurrentNetwork;
import machineLearning.neuralNetworks.SimpleRecurrentNetwork;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;
import visualization.VisualizeGraph;

public class TestRecurrentNeuralNetworks 
{
	
	public static void main(String[] args)
	{
		//testNormalNetwork();
		//testSimpleRecurrentNetwork();
		testGeneralRecurrentNetwork();
	}
	
	static void testSimpleRecurrentNetwork()
	{
		int numberExamples=50000;
		int exampleDuration=5;
		
		SimpleRecurrentNetwork srn=new SimpleRecurrentNetwork("simple recurrent network",
				new int[]{2, 1}, new int[]{1,1}, exampleDuration);
		
		
			
		generated=0;
		Matrix[][][] trainingInputs=new Matrix[numberExamples][][];
		Matrix[][][] trainingOutputs=new Matrix[numberExamples][][];
		for(int exampleInd=0; exampleInd<trainingInputs.length; exampleInd++)
		{
			trainingInputs[exampleInd]=new Matrix[exampleDuration][1];
			trainingOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				trainingInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				trainingInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				trainingInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				trainingOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				trainingOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
			}
		}
		
		generated=0;
		Matrix[][][] validationInputs=new Matrix[numberExamples/5][][];
		Matrix[][][] validationOutputs=new Matrix[numberExamples/5][][];
		for(int exampleInd=0; exampleInd<validationInputs.length; exampleInd++)
		{
			validationInputs[exampleInd]=new Matrix[exampleDuration][1];
			validationOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				validationInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				validationInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				validationInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				validationOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				validationOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
			}
		}
		
		new VisualizeGraph(srn.unrolledNetwork);
		
		ExampleBatchDerivativeOptimizer optimizer
			//=new RProp(null, null, 4000, 1000);
			//=new BackPropagation(null, null, 1000, 1000, 1.0f);
			=new Adam(null, null, 4000, 1000);
		
		srn.train(optimizer, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
	}
	
	static void testNormalNetwork()
	{
		ComputeGraph cg=new ComputeGraph("standard network");
		ComputeNode input=cg.addNode("input", new Input());
		
		ComputeNode hiddenWeights=cg.addNode("hidden1 weights", new ParamMMult(generateWeightMatrix(1, 2, 2)));
		ComputeNode hiddenBiases=cg.addNode("hidden1 biases", new ParamMAdd(generateBiasMatrix(1, 1)));
		ComputeNode hiddenSigmoid=cg.addNode("hidden1 sigmoid", new Sigmoid());

		/*
		ComputeNode outputWeights=cg.addNode("output weights", new MMult(generateWeightMatrix(1, 4, 4)));
		ComputeNode outputBiases=cg.addNode("output biases", new MAdd(generateBiasMatrix(1, 1)));
		ComputeNode outputSigmoid=cg.addNode("output sigmoid", new Sigmoid());
		*/
		ComputeNode trainingInput=cg.addNode("training outputs", new Input());
		ComputeNode objective=cg.addNode("euclideanCost", new Euclidean());
		
		input.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{hiddenWeights});
		
		hiddenWeights.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{hiddenBiases});
		hiddenBiases.setInputOutputNode(new ComputeNode[]{hiddenWeights}, new ComputeNode[]{hiddenSigmoid});
		hiddenSigmoid.setInputOutputNode(new ComputeNode[]{hiddenBiases}, new ComputeNode[]{objective});
		/*
		outputWeights.setInputOutputNode(new ComputeNode[]{hiddenSigmoid}, new ComputeNode[]{outputBiases});
		outputBiases.setInputOutputNode(new ComputeNode[]{outputWeights}, new ComputeNode[]{outputSigmoid});
		outputSigmoid.setInputOutputNode(new ComputeNode[]{outputBiases}, new ComputeNode[]{objective});
		*/
		trainingInput.setInputOutputNode(new ComputeNode[]{trainingInput}, new ComputeNode[]{objective});
		objective.setInputOutputNode(new ComputeNode[]{hiddenSigmoid, trainingInput}, new ComputeNode[]{objective});
		
		List<ComputeNode> outputVertices=new ArrayList<>();
		outputVertices.add(objective);
		cg.setOutputVertices(outputVertices);
		
		int numberExamples=100000;
		int exampleDuration=1;
		
		List<Hashtable<ComputeNode, Matrix>> trainingExamples=new ArrayList<>();
		List<Hashtable<ComputeNode, Matrix>> validationExamples=new ArrayList<>();
		
		Matrix[][][] trainingInputs=new Matrix[numberExamples][][];
		Matrix[][][] trainingOutputs=new Matrix[numberExamples][][];
		for(int exampleInd=0; exampleInd<trainingInputs.length; exampleInd++)
		{
			trainingInputs[exampleInd]=new Matrix[exampleDuration][1];
			trainingOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				trainingInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				trainingInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				trainingInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				trainingOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				trainingOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
				
				Hashtable<ComputeNode, Matrix> trainingInputsTable=new Hashtable<>();
				trainingInputsTable.put(input, trainingInputs[exampleInd][timeStep][0]);
				trainingInputsTable.put(trainingInput, trainingOutputs[exampleInd][timeStep][0]);
				trainingExamples.add(trainingInputsTable);
			}
			
		}
		
		Matrix[][][] validationInputs=new Matrix[numberExamples/5][][];
		Matrix[][][] validationOutputs=new Matrix[numberExamples/5][][];
		for(int exampleInd=0; exampleInd<validationInputs.length; exampleInd++)
		{
			validationInputs[exampleInd]=new Matrix[exampleDuration][1];
			validationOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				validationInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				validationInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				validationInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				validationOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				validationOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
				
				Hashtable<ComputeNode, Matrix> trainingInputsTable=new Hashtable<>();
				trainingInputsTable.put(input, trainingInputs[exampleInd][timeStep][0]);
				trainingInputsTable.put(trainingInput, trainingOutputs[exampleInd][timeStep][0]);
				validationExamples.add(trainingInputsTable);
			}
		}
		
		ExampleBatchDerivativeOptimizer optimizer
		//=new RProp(trainingExamples, validationExamples, 5000, 100);
		=new BackPropagation(trainingExamples, validationExamples, 5000, 100, 0.2f);
		optimizer.optimize(cg, new ComputeNode[]{objective});
	}
	
	static void testGeneralRecurrentNetwork()
	{
		int numberExamples=50000;
		int exampleDuration=2;
		
		int[] inputShape=new int[]{2, 1};
		int[] outputShape=new int[]{1,1};
		
		generated=0;
		Matrix[][][] trainingInputs=new Matrix[numberExamples][][];
		Matrix[][][] trainingOutputs=new Matrix[numberExamples][][];
		for(int exampleInd=0; exampleInd<trainingInputs.length; exampleInd++)
		{
			trainingInputs[exampleInd]=new Matrix[exampleDuration][1];
			trainingOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				trainingInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				trainingInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				trainingInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				trainingOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				trainingOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
			}
		}
		
		generated=0;
		Matrix[][][] validationInputs=new Matrix[numberExamples/5][][];
		Matrix[][][] validationOutputs=new Matrix[numberExamples/5][][];
		for(int exampleInd=0; exampleInd<validationInputs.length; exampleInd++)
		{
			validationInputs[exampleInd]=new Matrix[exampleDuration][1];
			validationOutputs[exampleInd]=new Matrix[exampleDuration][1];
			Matrix[] addData=generateAddData(exampleDuration);
			for(int timeStep=0; timeStep<exampleDuration; timeStep++)
			{
				validationInputs[exampleInd][timeStep][0]=new FMatrix(2, 1);
				validationInputs[exampleInd][timeStep][0].set(0, 0, addData[0].get(timeStep, 0));
				validationInputs[exampleInd][timeStep][0].set(1, 0, addData[1].get(timeStep, 0));
				validationOutputs[exampleInd][timeStep][0]=new FMatrix(1, 1);
				validationOutputs[exampleInd][timeStep][0].set(0, 0, addData[2].get(timeStep, 0));
			}
		}
		
		RecurrentComputeGraph lstm=new DeepLSTM("lstm", inputShape, outputShape);
				//new LSTM("lstm", inputShape, outputShape);
		
		//new VisualizeGraph(lstm);
		
		RecurrentNetwork rn=new RecurrentNetwork("General Recurrent Network", lstm, exampleDuration);
		
		//new VisualizeGraph(rn.unrolledNetwork);
		
		ExampleBatchDerivativeOptimizer optimizer
			=new RProp(null, null, 300, 1000);
			//=new BackPropagation(null, null, 30000, 1000, 0.1f);
			//=new Adam(null, null, 1000, 1000);
			//=new Nestrov(null, null, 1000, 1000, 0.1f, 0.95f);
		rn.train(optimizer, trainingInputs, trainingOutputs, validationInputs, validationOutputs);
	}
	
	static Random rand=new Random(52341);
	static int generated;
	
	private static Matrix[] generateAddData(int length)
	{
		if(generated==5474)
		{
			int u=0;
		}
		Matrix add0=new FMatrix(length, 1);
		Matrix add1=new FMatrix(length, 1);
		Matrix sum=new FMatrix(length, 1);
		
		int carry=0;
		for(int ind=0; ind<sum.getLen(); ind++)
		{
			add0.set(ind, 0, (int)Math.round(rand.nextFloat()));
			add1.set(ind, 0, (int)Math.round(rand.nextFloat()));
			int subSum=(int)(add0.get(ind, 0)+add1.get(ind, 0));
			
			/*
			if(subSum+carry>0)
			{
				sum.set(ind, 0, 1);
			}
			else
			{
				sum.set(ind, 0, 0);
			}
			*/
			//carry=0;
			if(subSum+carry<2)
			{
				sum.set(ind, 0, subSum+carry);
				carry=0;
			}
			else if(subSum+carry==2)
			{
				sum.set(ind, 0, 0);
				carry=1;
			}
			else if(subSum+carry==3)
			{
				sum.set(ind, 0, 1);
				carry=1;
			}
		}
		generated++;
		return new Matrix[]{add0, add1, sum};
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
