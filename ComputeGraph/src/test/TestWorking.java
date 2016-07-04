package test;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import graph.ComputeGraph;
import machineLearning.Learner.BackPropagation;
import machineLearning.Learner.RProp;
import machineLearning.activationFunction.Sigmoid;
import machineLearning.activationFunction.TanH;
import machineLearning.costFunction.Euclidean;
import machineLearning.generalFunctions.Input;
import machineLearning.matrixTransform.Combine;
import machineLearning.matrixTransform.ParamMAdd;
import machineLearning.matrixTransform.ParamMMult;
import machineLearning.matrixTransform.Split;
import matrix.FMatrix;
import matrix.Matrix;
import vertex.ComputeNode;

public class TestWorking 
{
	
	public static void main(String[] args) throws IOException
	{
		//testNetworkOutput();
		//testMNIST();
		testMNISTSplit();
	}
	
	static void testNetworkOutput()
	{
		ComputeGraph cg=new ComputeGraph("standard network");
		cg.addNode("input", new Input());
		cg.addNode("hidden1 weights", new ParamMMult(new FMatrix(new float[][]{new float[]{0.15f, 0.20f}, new float[]{0.25f, 0.30f}})));
		cg.addNode("hidden1 biases", new ParamMAdd(new FMatrix(new float[][]{new float[]{0.35f}, new float[]{0.35f}})));
		cg.addNode("hidden1 sigmoid", new Sigmoid());
		cg.addNode("output weights", new ParamMMult(new FMatrix(new float[][]{new float[]{0.40f, 0.45f}, new float[]{0.50f, 0.55f}})));
		cg.addNode("output biases", new ParamMAdd(new FMatrix(new float[][]{new float[]{0.60f}, new float[]{0.60f}})));
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
		inputs.put("training outputs", new FMatrix(new float[][]{new float[]{0.01f}, new float[]{0.99f}}));
		Matrix output=cg.getOutput(inputs).get("output sigmoid");
		
		outputVertices=new ArrayList<>();
		outputVertices.add("cost function");
		cg.setOutputVertices(outputVertices);
		
		Hashtable<String, Matrix>[] derivatives=cg.derive(inputs);
		Hashtable<String, Matrix> objectiveDerivatives=derivatives[0];
		Hashtable<String, Matrix> parameterDerivatives=derivatives[1];
		int u=0;
	}
	
	static void testMNIST() throws IOException
	{
		ComputeGraph cg=new ComputeGraph("standard network");
		ComputeNode input=cg.addNode("input", new Input());
		ComputeNode weights1=cg.addNode("hidden1 weights", new ParamMMult(generateWeightMatrix(100, 28*28, 28*28)));
		ComputeNode biases1=cg.addNode("hidden1 biases", new ParamMAdd(generateBiasMatrix(100, 1)));
		ComputeNode sigmoid1=cg.addNode("hidden1 sigmoid", new TanH());
		ComputeNode outWeights=cg.addNode("output weights", new ParamMMult(generateWeightMatrix(10, 100, 100)));
		ComputeNode outBiases=cg.addNode("output biases", new ParamMAdd(generateBiasMatrix(10, 1)));
		ComputeNode outSigmoid=cg.addNode("output sigmoid", new TanH());
		ComputeNode trainingOutputs=cg.addNode("training outputs", new Input());
		ComputeNode cost=cg.addNode("cost function", new Euclidean());
		
		input.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{weights1});
		weights1.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{biases1});
		biases1.setInputOutputNode(new ComputeNode[]{weights1}, new ComputeNode[]{sigmoid1});
		sigmoid1.setInputOutputNode(new ComputeNode[]{biases1}, new ComputeNode[]{outWeights});
		outWeights.setInputOutputNode(new ComputeNode[]{sigmoid1}, new ComputeNode[]{outBiases});
		outBiases.setInputOutputNode(new ComputeNode[]{outWeights}, new ComputeNode[]{outSigmoid});
		outSigmoid.setInputOutputNode(new ComputeNode[]{outBiases}, new ComputeNode[]{cost});		
		cost.setInputOutputNode(new ComputeNode[]{outSigmoid, trainingOutputs}, new ComputeNode[]{cost});
		
		trainingOutputs.setInputOutputNode(new ComputeNode[]{trainingOutputs}, new ComputeNode[]{cost});
		
		Object[] training=getImagesAndLabels("train");
		float[][] trainImages=(float[][])training[0];
		float[][] trainLabels=(float[][])training[1];
		List<Hashtable<ComputeNode, Matrix>> trainingInputsList=new ArrayList<>();
		for(int inputInd=0; inputInd<trainImages.length; inputInd++)
		{
			Hashtable<ComputeNode, Matrix> trainingInputs=new Hashtable<>();
			trainingInputs.put(input, new FMatrix(new float[][]{trainImages[inputInd]}).otrans());
			trainingInputs.put(trainingOutputs, new FMatrix(new float[][]{trainLabels[inputInd]}).otrans());
			trainingInputsList.add(trainingInputs);
		}
		Object[] eval=getImagesAndLabels("t10k");
		float[][] validationImages=(float[][])eval[0];
		float[][] validationLabels=(float[][])eval[1];
		List<Hashtable<ComputeNode, Matrix>> validationInputsList=new ArrayList<>();	
		for(int inputInd=0; inputInd<validationImages.length; inputInd++)
		{
			Hashtable<ComputeNode, Matrix> validationInputs=new Hashtable<>();
			validationInputs.put(input, new FMatrix(new float[][]{validationImages[inputInd]}).otrans());
			validationInputs.put(trainingOutputs, new FMatrix(new float[][]{validationLabels[inputInd]}).otrans());
			validationInputsList.add(validationInputs);
		}
		
		List<ComputeNode> outputVertices=new ArrayList<>();
		outputVertices.add(cost);
		cg.setOutputVertices(outputVertices);

		BackPropagation bProp=new BackPropagation(trainingInputsList, 
				validationInputsList,
				100, 50, 0.1f);
		
		bProp.optimize(cg, cost);
	}
	
	static void testMNISTSplit() throws IOException
	{
		ComputeGraph cg=new ComputeGraph("standard network");
		ComputeNode input=cg.addNode("input", new Input());
		ComputeNode weights1=cg.addNode("hidden1 weights", new ParamMMult(generateWeightMatrix(100, 28*28, 28*28)));
		ComputeNode biases1=cg.addNode("hidden1 biases", new ParamMAdd(generateBiasMatrix(100, 1)));
		ComputeNode sigmoid1=cg.addNode("hidden1 sigmoid", new TanH());
		
		ComputeNode split=cg.addNode("split", new Split(50, 0));
		
		ComputeNode weights2a=cg.addNode("hidden2a weights", new ParamMMult(generateWeightMatrix(20, 50, 50)));
		ComputeNode biases2a=cg.addNode("hidden2a biases", new ParamMAdd(generateBiasMatrix(20, 1)));
		ComputeNode sigmoid2a=cg.addNode("hidden2a sigmoid", new TanH());
		
		ComputeNode weights2b=cg.addNode("hidden2b weights", new ParamMMult(generateWeightMatrix(20, 50, 50)));
		ComputeNode biases2b=cg.addNode("hidden2b biases", new ParamMAdd(generateBiasMatrix(20, 1)));
		ComputeNode sigmoid2b=cg.addNode("hidden2b sigmoid", new TanH());
		
		ComputeNode combine=cg.addNode("combine", new Combine(20, 0));
		
		ComputeNode outWeights=cg.addNode("output weights", new ParamMMult(generateWeightMatrix(10, 40, 40)));
		ComputeNode outBiases=cg.addNode("output biases", new ParamMAdd(generateBiasMatrix(10, 1)));
		ComputeNode outSigmoid=cg.addNode("output sigmoid", new TanH());
		ComputeNode trainingOutputs=cg.addNode("training outputs", new Input());
		ComputeNode cost=cg.addNode("cost function", new Euclidean());
		
		input.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{weights1});
		weights1.setInputOutputNode(new ComputeNode[]{input}, new ComputeNode[]{biases1});
		biases1.setInputOutputNode(new ComputeNode[]{weights1}, new ComputeNode[]{sigmoid1});
		sigmoid1.setInputOutputNode(new ComputeNode[]{biases1}, new ComputeNode[]{split});
		
		split.setInputOutputNode(new ComputeNode[]{sigmoid1}, new ComputeNode[]{weights2a, weights2b});
		
		weights2a.setInputOutputNode(new ComputeNode[]{split}, new ComputeNode[]{biases2a});
		biases2a.setInputOutputNode(new ComputeNode[]{weights2a}, new ComputeNode[]{sigmoid2a});
		sigmoid2a.setInputOutputNode(new ComputeNode[]{biases2a}, new ComputeNode[]{combine});
		
		weights2b.setInputOutputNode(new ComputeNode[]{split}, new ComputeNode[]{biases2b});
		biases2b.setInputOutputNode(new ComputeNode[]{weights2b}, new ComputeNode[]{sigmoid2b});
		sigmoid2b.setInputOutputNode(new ComputeNode[]{biases2b}, new ComputeNode[]{combine});
		
		combine.setInputOutputNode(new ComputeNode[]{sigmoid2a, sigmoid2b}, new ComputeNode[]{outWeights});
		
		outWeights.setInputOutputNode(new ComputeNode[]{combine}, new ComputeNode[]{outBiases});
		outBiases.setInputOutputNode(new ComputeNode[]{outWeights}, new ComputeNode[]{outSigmoid});
		outSigmoid.setInputOutputNode(new ComputeNode[]{outBiases}, new ComputeNode[]{cost});		
		cost.setInputOutputNode(new ComputeNode[]{outSigmoid, trainingOutputs}, new ComputeNode[]{cost});
		
		trainingOutputs.setInputOutputNode(new ComputeNode[]{trainingOutputs}, new ComputeNode[]{cost});
		
		Object[] training=getImagesAndLabels("train");
		float[][] trainImages=(float[][])training[0];
		float[][] trainLabels=(float[][])training[1];
		List<Hashtable<ComputeNode, Matrix>> trainingInputsList=new ArrayList<>();
		for(int inputInd=0; inputInd<trainImages.length; inputInd++)
		{
			Hashtable<ComputeNode, Matrix> trainingInputs=new Hashtable<>();
			trainingInputs.put(input, new FMatrix(new float[][]{trainImages[inputInd]}).otrans());
			trainingInputs.put(trainingOutputs, new FMatrix(new float[][]{trainLabels[inputInd]}).otrans());
			trainingInputsList.add(trainingInputs);
		}
		Object[] eval=getImagesAndLabels("t10k");
		float[][] validationImages=(float[][])eval[0];
		float[][] validationLabels=(float[][])eval[1];
		List<Hashtable<ComputeNode, Matrix>> validationInputsList=new ArrayList<>();	
		for(int inputInd=0; inputInd<validationImages.length; inputInd++)
		{
			Hashtable<ComputeNode, Matrix> validationInputs=new Hashtable<>();
			validationInputs.put(input, new FMatrix(new float[][]{validationImages[inputInd]}).otrans());
			validationInputs.put(trainingOutputs, new FMatrix(new float[][]{validationLabels[inputInd]}).otrans());
			validationInputsList.add(validationInputs);
		}
		
		List<ComputeNode> outputVertices=new ArrayList<>();
		outputVertices.add(cost);
		cg.setOutputVertices(outputVertices);

		BackPropagation bProp=new BackPropagation(trainingInputsList, 
				validationInputsList,
				100, 50, 0.1f);
		
		bProp.optimize(cg, cost);
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
	
	static Object[] getImagesAndLabels(String fileName) throws IOException
	{
	    BufferedInputStream labels = new BufferedInputStream(new FileInputStream("/home/c/workspace2/mlGPU/data/MNIST_Numbers/"+fileName+"-labels.idx1-ubyte"));
		BufferedInputStream images = new BufferedInputStream(new FileInputStream("/home/c/workspace2/mlGPU/data/MNIST_Numbers/"+fileName+"-images.idx3-ubyte"));
	    
		byte[] intBytes=new byte[4];
		labels.read(intBytes);
		int magicNumber = b8ToB10Int(intBytes);
	    if (magicNumber != 2049) {
	      System.err.println("Label file has wrong magic number: " + magicNumber + " (should be 2049)");
	      System.exit(0);
	    }
	    images.read(intBytes);
	    magicNumber = b8ToB10Int(intBytes);
	    if (magicNumber != 2051) {
	      System.err.println("Image file has wrong magic number: " + magicNumber + " (should be 2051)");
	      System.exit(0);
	    }
	    labels.read(intBytes);
	    int numLabels = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numImages = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numRows = b8ToB10Int(intBytes);
	    images.read(intBytes);
	    int numCols = b8ToB10Int(intBytes);
	    if (numLabels != numImages) {
	      System.err.println("Image file and label file do not contain the same number of entries.");
	      System.err.println("  Label file contains: " + numLabels);
	      System.err.println("  Image file contains: " + numImages);
	      System.exit(0);
	    }

	    long start = System.currentTimeMillis();
	    int numLabelsRead = 0;
	    int numImagesRead = 0;
	    float[][] imagesArray=new float[numImages][28*28];
	    float[][] labelsArray=new float[numImages][10];
	    byte[] labelsBytes=new byte[numImages];
	    labels.read(labelsBytes);
	    byte[] imageBytes=new byte[numImages*28*28];
	    images.read(imageBytes);
	    while (numLabelsRead < numLabels) 
	    {
	    	labelsArray[numLabelsRead][labelsBytes[numLabelsRead]]=1.0f;
	    	numLabelsRead++;
	      for (int colIdx = 0; colIdx < numCols; colIdx++) 
	      {
	        for (int rowIdx = 0; rowIdx < numRows; rowIdx++) 
	        {
	        	imagesArray[numImagesRead][28*colIdx+rowIdx]=(float)(Byte.toUnsignedInt(imageBytes[28*28*numImagesRead+28*colIdx+rowIdx]))/256;
	        }
	      }
	      float[] image=imagesArray[0];
	      numImagesRead++;
	    }
	    
	    long end = System.currentTimeMillis();
	    long elapsed = end - start;
	    System.out.println("Read " + numLabelsRead + " samples in " + elapsed + " ms ");
	    return new Object[]{imagesArray, labelsArray};
		
	}
	
	static int b8ToB10Int(byte[] b8)
    {
        return ((b8[0] & 0xFF) << 24) |
         ((b8[1] & 0xFF) << 16) |
         ((b8[2] & 0xFF) <<  8) |
         (b8[3] & 0xFF);
    }
	
}
