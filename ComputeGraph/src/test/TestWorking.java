package test;

import org.apache.commons.math3.distribution.NormalDistribution;

import activationFunctions.RectifiedLinearActivationFunction;
import graph.ComputeGraph;
import machineLearning.generalFunctions.Input;
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
		ComputeGraph cg=new ComputeGraph("standard network");
		cg.addNode("input", new Input());
	}
	
	static Matrix generateWeightMatrix(int rows, int cols, int outputSize)
	{
		Matrix weights=new FMatrix(rows, cols);
		NormalDistribution nInvGaussian=new NormalDistribution(0.0, 1.0/Math.sqrt(outputSize));
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
