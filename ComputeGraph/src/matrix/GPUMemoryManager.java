package matrix;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.Hashtable;

import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.runtime.cudaError.cudaErrorMemoryAllocation;

public class GPUMemoryManager 
{
	
	public static volatile Hashtable<Pointer, Integer> pointerUsages=new Hashtable<>();
	protected static Pointer blockPointer;
	
	public static void init(long amount)
	{
		if(FMatrix.GPU)
		{
			int res=cudaMalloc(blockPointer, amount);
			if(res==cudaErrorMemoryAllocation)
			{
				System.out.println("Error: Out of GPU memory!");
			}
		}
	}
	
	public static Pointer alloc(long amount)
	{
		Pointer pointer=new Pointer();
		int res=cudaMalloc(pointer, amount);
		if(res==cudaErrorMemoryAllocation)
		{
			System.out.println("Error: Out of GPU memory!");
		}
		synchronized(pointerUsages)
		{
			if(pointerUsages.get(pointer)!=null)
			{
				Pointer pointer2=new Pointer();
				cudaMalloc(pointer2, amount);
				int u=0;
			}
			pointerUsages.put(pointer, 1);
		}
		return pointer;
	}
	
	public static void incUsages(Pointer pointer)
	{
		synchronized(pointerUsages)
		{
			if(pointerUsages.get(pointer)==null)
			{
				pointerUsages.put(pointer, 1);
			}
			else
			{
				pointerUsages.put(pointer, pointerUsages.get(pointer)+1);
			}
		}
	}
	
	public static void free(Pointer pointer)
	{
		if(pointer!=null)
		{
			synchronized(pointerUsages)
			{
				int usages=0;
				try
				{
					usages=pointerUsages.get(pointer)-1;
				}
				catch(Exception e)
				{			
					e.printStackTrace();
				}
				if(usages>0)
				{
					pointerUsages.put(pointer, usages);
				}
				else
				{
					pointerUsages.remove(pointer);
					cudaFree(pointer);
				}
			}
		}
	}

}
