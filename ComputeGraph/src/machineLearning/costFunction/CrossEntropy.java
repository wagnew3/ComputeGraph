package machineLearning.costFunction;

import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import matrix.FMatrix;
import matrix.Matrix;
import function.DifferentiableFunction;

public class CrossEntropy  extends DifferentiableFunction
{
	
	public static CUcontext context;
    private static CUmodule module;
    private static CUfunction crossEntropyCost;
    private static CUfunction crossEntropyCostDerivative;
    private static CUdeviceptr deviceBuffer;

    static
    {
    	if(FMatrix.GPU)
    	{
	    	JCudaDriver.setExceptionsEnabled(true);
	        init();
    	}
    }

	@Override
	public Matrix[][] differentiate(Matrix[] input, Matrix[] dInput) 
	{
		Matrix results=new FMatrix(input[0].getRows(), input[0].getCols());
		if(FMatrix.GPU)
		{
			((FMatrix)input[1]).sendToGPU();
			((FMatrix)input[0]).sendToGPU();
			
			int maxThreads=128;
	    	int maxBlocks=64;
	    	int numBlocks = getNumBlocks(input[1].getLen(), maxBlocks, maxThreads);
	        int numThreads = getNumThreads(input[1].getLen(), maxBlocks, maxThreads);
	        
	        int sharedMemSize = numThreads * Sizeof.FLOAT;
	        if (numThreads <= 32) 
	        {
	            sharedMemSize *= 2;
	        }
	        
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(((FMatrix)input[1]).gpuPointer),
	            Pointer.to(new int[]{input[1].getLen()}),
	            Pointer.to(((FMatrix)input[0]).gpuPointer),
	            Pointer.to(((FMatrix)results).gpuPointer)
	        );

	        // Call the kernel function.
	        cuLaunchKernel(crossEntropyCostDerivative,
	            numBlocks,  1, 1,         // Grid dimension
	            numThreads, 1, 1,         // Block dimension
	            sharedMemSize, null,   // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
			return new Matrix[][]{new Matrix[]{results}, null};
		}
		else
		{
			for(int costInd=0; costInd<input[0].getLen(); costInd++)
			{
				results.set(costInd, 0, 
						-((0.00001f+input[1].get(costInd, 0))/((input[0].get(costInd, 0))+0.00001f)
						-(1.00001f-input[1].get(costInd, 0))/(1.00001f-input[0].get(costInd, 0))));
				if(Float.isNaN(-((0.00001f+input[1].get(costInd, 0))/((input[0].get(costInd, 0))+0.00001f)
						-(1.00001f-input[1].get(costInd, 0))/(1.00001f-input[0].get(costInd, 0)))))
				{
					int u=0;
				}
			}
			return new Matrix[][]{new Matrix[]{results}, null};
		}
	}

	@Override
	public Matrix[] apply(Matrix[] input) 
	{
		if(FMatrix.GPU)
    	{
	    	float cost=0.0f;
			((FMatrix)input[1]).sendToGPU();
			((FMatrix)input[0]).sendToGPU();
			
			int maxThreads=128;
	    	int maxBlocks=64;
	    	int numBlocks = getNumBlocks(input[1].getLen(), maxBlocks, maxThreads);
	        int numThreads = getNumThreads(input[1].getLen(), maxBlocks, maxThreads);
	        
	        int sharedMemSize = numThreads * Sizeof.FLOAT;
	        if (numThreads <= 32) 
	        {
	            sharedMemSize *= 2;
	        }
	        
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(((FMatrix)input[1]).gpuPointer),
	            Pointer.to(new int[]{input[1].getLen()}),
	            Pointer.to(((FMatrix)input[0]).gpuPointer),
	            Pointer.to(((FMatrix)input[0]).gpuPointer)
	        );

	        // Call the kernel function.
	        cuLaunchKernel(crossEntropyCost,
	            numBlocks,  1, 1,         // Grid dimension
	            numThreads, 1, 1,         // Block dimension
	            sharedMemSize, null,   // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cost+=((FMatrix)input[0]).getSum();
			return new Matrix[]{new FMatrix(new float[][]{new float[]{-cost}})};
    	}
    	else
    	{
	    	float cost=0.0f;
			for(int costInd=0; costInd<input[0].getLen(); costInd++)
			{
				cost+=input[1].get(costInd, 0)*Math.log(input[0].get(costInd, 0)+0.0001f)
						+(1.0001f-input[1].get(costInd, 0))*Math.log(1.0001f-input[0].get(costInd, 0));
			}
			return new Matrix[]{new FMatrix(new float[][]{new float[]{-cost}})};
    	}
	}
	
	private static void init()
    {
        //context = Sigmoid.context;
        prepare();
    }
    
    public static void prepare()
    {
        // Prepare the ptx file.
        String ptxSigmoidFileName = null;
        try
        {
        	ptxSigmoidFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/costFunctions/crossEntropyCost.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        String ptxSigmoidDerivativeFileName = null;
        try
        {
        	ptxSigmoidDerivativeFileName=preparePtxFile(Matrix.workspaceDir+"mlGPU/src/costFunctions/crossEntropyCostDerivative.cu");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Could not prepare PTX file", e);
        }
        
        // Load the module from the PTX file
        module=new CUmodule();
        
        // Obtain a function pointer to the "reduce" function.
        cuModuleLoad(module, ptxSigmoidFileName);
        crossEntropyCost=new CUfunction();
        cuModuleGetFunction(crossEntropyCost, module, "crossEntropyCost");
        
        cuModuleLoad(module, ptxSigmoidDerivativeFileName);
        crossEntropyCostDerivative=new CUfunction();
        cuModuleGetFunction(crossEntropyCostDerivative, module, "crossEntropyCostDerivative");
    }
    
    public static void shutdown()
    {
        cuModuleUnload(module);
        if(deviceBuffer!=null)
        {
        	cuMemFree(deviceBuffer);
        }
        if (context != null)
        {
            cuCtxDestroy(context);
        }
    }
    
    private static int getNumBlocks(int n, int maxBlocks, int maxThreads)
    {
        int blocks = 0;
        int threads = getNumThreads(n, maxBlocks, maxThreads);
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
        blocks = Math.min(maxBlocks, blocks);
        return blocks;
    }

    private static int getNumThreads(int n, int maxBlocks, int maxThreads)
    {
        int threads = 0;
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        return threads;
    }
    
    private static int nextPow2(int x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    private static float[] createRandomArray(int size)
    {
        Random random = new Random();
        float array[] = new float[size];
        for(int i = 0; i < size; i++)
        {
            array[i] = random.nextFloat();
        }
        return array;
    }
    
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "/usr/local/cuda/bin/nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

}
