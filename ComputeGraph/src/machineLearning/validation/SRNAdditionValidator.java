package machineLearning.validation;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import machineLearning.neuralNetworks.SimpleRecurrentNetwork;
import matrix.Matrix;
import vertex.ComputeNode;

public class SRNAdditionValidator extends Validator
{
	
	SimpleRecurrentNetwork srn;
	List<Hashtable<ComputeNode, Matrix>> validationExamples;
	ComputeNode[] objectives;
	
	public SRNAdditionValidator(SimpleRecurrentNetwork srn, 
			List<Hashtable<ComputeNode, Matrix>> validationExamples,
			ComputeNode[] objectives)
	{
		this.srn=srn;
		this.validationExamples=validationExamples;
		this.objectives=objectives;
	}

	@Override
	public float validate() 
	{
		float totalError=0.0f;
		
		int numberCorrect=0;
		long numberErrors=0;
		for(int objectiveInd=0; objectiveInd<objectives.length; objectiveInd++)
		{
			for(Hashtable<ComputeNode, Matrix> validationExample: validationExamples)
			{
				Hashtable<ComputeNode, Hashtable<ComputeNode, Matrix>> output=srn.unrolledNetwork.compute(validationExample);
				if(output.get(objectives[objectiveInd])!=null)
				{
					totalError+=output.get(objectives[objectiveInd]).get(objectives[objectiveInd]).get(0, 0);
					numberErrors++;
				}
				/*
				float max=-1;
				int maxInd=-1;
				for(int outInd=0; outInd<output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).getLen(); outInd++)
				{
					if(max<output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).get(outInd, 0))
					{
						max=output.get(objectives[objectiveInd]).get(objectives[objectiveInd].inputNodes[0]).get(outInd, 0);
						maxInd=outInd;
					}
				}
				int correctInd=-2;
				for(int outInd=0; outInd<validationExample.get(objectives[objectiveInd].inputNodes[1]).getLen(); outInd++)
				{
					if(validationExample.get(objectives[objectiveInd].inputNodes[1]).get(outInd, 0)==1.0f)
					{
						correctInd=outInd;
						break;
					}
				}
				if(maxInd!=correctInd)
				{
					numberCorrect++;
				}
				*/
			}
		}
		
		System.out.println("Number classified incorrectly: "+numberCorrect);
		
		totalError/=numberErrors;
		return totalError;
	}

}
