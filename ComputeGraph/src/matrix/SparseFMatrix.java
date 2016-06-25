package matrix;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxDestroy;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleUnload;
import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasScopy;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.runtime.JCuda.cudaMemset;

import static jcuda.jcusparse.JCusparse.*;
import static jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO;
import static jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.Random;

import activationFunctions.Sigmoid;
import jcuda.*;
import jcuda.jcusparse.*;
import jcuda.runtime.JCuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasHandle;

public class SparseFMatrix extends Matrix
{
	
	protected int rows;
	protected int columns;
	public float[] values;
	public int[] rowIndices;
	public int[] colIndices;
	
	protected int nnz;
	
	protected boolean inGPU;
	public Pointer valuesPointer;
	public Pointer rowsIndicesPointer;
	public Pointer colsIndicesPointer;
	protected float sum;
	protected static cusparseHandle handle;
	public static cusparseMatDescr descra;
	
	static
	{
		handle=new cusparseHandle();
		cusparseCreate(handle);
		descra=new cusparseMatDescr();cusparseCreateMatDescr(descra);
        cusparseSetMatType(descra, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descra, CUSPARSE_INDEX_BASE_ZERO);
	}
	
	private SparseFMatrix()
	{
		
	}
	
	public SparseFMatrix(float[] values, int[] rowIndices, int[] colIndices, int rows, int cols)
	{
		this.rows=rows;
		this.columns=cols;
		nnz=values.length;
		this.values=values;
		this.rowIndices=rowIndices;
		this.colIndices=colIndices;
		inGPU=false;
		valuesPointer=null;
		rowsIndicesPointer=null;
		colsIndicesPointer=null;
		
		sum=0.0f;
		for(int valInd=0; valInd<values.length; valInd++)
		{
			sum+=values[valInd];
		}
	}

	public void sendToGPU()
	{
		if(!inGPU)
		{
			if(valuesPointer==null)
			{
				valuesPointer=GPUMemoryManager.alloc(nnz*Sizeof.FLOAT);
				rowsIndicesPointer=GPUMemoryManager.alloc(nnz*Sizeof.INT);
				colsIndicesPointer=GPUMemoryManager.alloc(nnz*Sizeof.INT);
			}
			
			inGPU=true;
	        
			cublasSetVector(nnz, Sizeof.FLOAT, Pointer.to(values), 1, valuesPointer, 1);
			cublasSetVector(nnz, Sizeof.INT, Pointer.to(rowIndices), 1, rowsIndicesPointer, 1);
			cublasSetVector(nnz, Sizeof.INT, Pointer.to(colIndices), 1, colsIndicesPointer, 1);
			
			Pointer csrRowsIndicesPointer=GPUMemoryManager.alloc((getRows()+1)*Sizeof.INT);
			cusparseXcoo2csr(handle, rowsIndicesPointer, nnz, getRows(),
					csrRowsIndicesPointer, CUSPARSE_INDEX_BASE_ZERO);
			
			GPUMemoryManager.free(rowsIndicesPointer);
			rowsIndicesPointer=csrRowsIndicesPointer;

			values=null;
			rowIndices=null;
			colIndices=null;
		}
	}
	
	public void getFromGPU()
	{
		if(inGPU)
		{
			if(nnz==0)
			{
				values=new float[0];
				rowIndices=new int[0];
				colIndices=new int[0];
			}
			else
			{
				Pointer cooRowsIndicesPointer=GPUMemoryManager.alloc(nnz*Sizeof.INT);
				cusparseXcsr2coo(handle, rowsIndicesPointer, nnz, getRows(),
						cooRowsIndicesPointer, CUSPARSE_INDEX_BASE_ZERO);
				
				GPUMemoryManager.free(rowsIndicesPointer);
				rowsIndicesPointer=cooRowsIndicesPointer;
				
				values=new float[nnz];
				rowIndices=new int[nnz];
				colIndices=new int[nnz];
				
				cublasGetVector(nnz, Sizeof.FLOAT, valuesPointer, 1, Pointer.to(values), 1);
				cublasGetVector(nnz, Sizeof.INT, rowsIndicesPointer, 1, Pointer.to(rowIndices), 1);
				cublasGetVector(nnz, Sizeof.INT, colsIndicesPointer, 1, Pointer.to(colIndices), 1);
				
				for(int colInd=0; colInd<colIndices.length; colInd++)
				{
					if(colIndices[colInd]<0)
					{
						int u=0;
					}
				}
				
				GPUMemoryManager.free(valuesPointer);
				GPUMemoryManager.free(rowsIndicesPointer);
				GPUMemoryManager.free(colsIndicesPointer);
				
				
			}
			
			valuesPointer=null;
			rowsIndicesPointer=null;
			colsIndicesPointer=null;
	        inGPU=false;
		}
	}
	
	public void getFromGPUNoFree()
	{
		if(inGPU)
		{
			if(nnz==0)
			{
				values=new float[0];
				rowIndices=new int[0];
				colIndices=new int[0];
			}
			else
			{
				Pointer cooRowsIndicesPointer=GPUMemoryManager.alloc(nnz*Sizeof.INT);
				cusparseXcsr2coo(handle, rowsIndicesPointer, nnz, getRows(),
						cooRowsIndicesPointer, CUSPARSE_INDEX_BASE_ZERO);
				
				GPUMemoryManager.free(rowsIndicesPointer);
				rowsIndicesPointer=cooRowsIndicesPointer;
				
				values=new float[nnz];
				rowIndices=new int[nnz];
				colIndices=new int[nnz];
				
				cublasGetVector(nnz, Sizeof.FLOAT, valuesPointer, 1, Pointer.to(values), 1);
				cublasGetVector(nnz, Sizeof.INT, rowsIndicesPointer, 1, Pointer.to(rowIndices), 1);
				cublasGetVector(nnz, Sizeof.INT, colsIndicesPointer, 1, Pointer.to(colIndices), 1);
				
				for(int colInd=0; colInd<colIndices.length; colInd++)
				{
					if(colIndices[colInd]<0)
					{
						int u=0;
					}
				}
			}
	        inGPU=false;
		}
	}
	
	@Override
	public void clear()
	{
		if(inGPU)
		{
			GPUMemoryManager.free(valuesPointer);
			GPUMemoryManager.free(rowsIndicesPointer);
			GPUMemoryManager.free(colsIndicesPointer);
			valuesPointer=null;
			rowsIndicesPointer=null;
			colsIndicesPointer=null;
	        inGPU=false;
		}
	}
	
	protected void setRowPointer(Pointer rowPointer)
	{
		if(rowPointer!=null
				&& !rowPointer.equals(rowsIndicesPointer))
		{
			GPUMemoryManager.free(rowsIndicesPointer);
			GPUMemoryManager.incUsages(rowPointer);
		}		
		rowsIndicesPointer=rowPointer;
	}
	
	@Override
	public void finalize()
	{
		clear();
	}

	@Override
	public int getRows() 
	{
		return rows;
	}

	@Override
	public int getCols() 
	{
		return columns;
	}

	@Override
	public int getLen() 
	{
		return rows*columns;
	}
	
	public int getNNZ()
	{
		return nnz;
	}

	@Override
	public float get(int row, int col) 
	{
		getFromGPU();
		for(int nzInd=0; nzInd<nnz && rowIndices[nzInd]<=row; nzInd++)
		{
			if(rowIndices[nzInd]==row && colIndices[nzInd]==col)
			{
				return values[nzInd];
			}
		}
		return 0.0f;
	}

	@Override
	public void set(int row, int col, float val) 
	{
		getFromGPU();
		System.out.println("Error: Spare matrix setting unimplemented!");
	}
	
	@Override
	public  float[] getData()
	{
		getFromGPU();
		System.out.println("Error: Spare matrix getData() unimplemented!");
		return null;
	}
	
	@Override
	public Matrix getSubVector(int offset, int length)
	{
		System.out.println("Error: Spare matrix getSubVector() unimplemented!");
		return null;
	}
	
	@Override
	public Matrix otrans() 
	{
		if(!(rows==1 || columns==1))
		{
			System.out.println("transpose not yet implemented for non-vectors!");
		}
		int tempColumns=columns;
		columns=rows;
		rows=tempColumns;
		return this;
	}
	
	@Override
	public Matrix append(Matrix toAppend)
	{
		System.out.println("Error: Spare matrix append() unimplemented!");
		return null;
	}

	@Override
	public Matrix mmult(Matrix toMultiplyBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix ommult(Matrix toMultiplyBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix oebemult(Matrix multVec) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix mscal(float toScaleBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omscal(float toScaleBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix mad(Matrix toAddTo) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omad(Matrix toAddTo) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix madScale(Matrix toAddTo, float scaleAddBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omadScale(Matrix toAddTo, float scaleAddBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix msub(Matrix toSubtractBy, Matrix result) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omsub(Matrix toSubtractBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix msubScale(Matrix toSubtractBy, float scaleSubBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omsubScale(Matrix toSubtractBy, float scaleSubBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float dot(Matrix toDotWith) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public Matrix matVecMultScale(Matrix mat, Matrix vec, float scaleSubBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omatVecMultScale(Matrix mat, Matrix vec, float scaleSubBy) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix matVecMultScaleAdd(Matrix mat, Matrix vecMult, float scaleSubBy, Matrix vecAdd)
	{
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix omatVecMultScaleAdd(Matrix mat, Matrix vecMult, float scaleMultBy, Matrix vecAdd, Matrix result,
			float scaleResultBy) 
	{
		/*
		if(((SparseFMatrix)mat).getNNZ()>0)
		{
			csrmv(false, scaleMultBy, (SparseFMatrix)mat, (FDMatrix)vecMult, scaleResultBy, (FDMatrix)result);
		}
		*/
		
		if(((SparseFMatrix)mat).getNNZ()>0)
		{
			csrmm(0, 0, scaleMultBy, (SparseFMatrix)mat, (FMatrix)vecMult,
					0.0f, (FMatrix)result);
		}
		return ((FMatrix)result).saxpy(1.0f, vecAdd, 1, result, 1, result);
	}

	@Override
	public Matrix outProd(Matrix vecA, Matrix vecB, Matrix result) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix sgemm(boolean transposeA, boolean transposeB, Matrix toMultiplyBy, float alpha, float beta,
			Matrix toAdd, Matrix result) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Matrix matVecMultScaleAddScale(Matrix mat, Matrix vecMult, float scaleSubBy, Matrix vecAdd,
			float scaleAddBy) 
	{
		return null;
	}

	@Override
	public Matrix scal(float alpha, int incx, Matrix result) 
	{
		getData();
		result=new SparseFMatrix(values.clone(), colIndices.clone(), rowIndices.clone(), getRows(), getCols());
		for(int rowInd=0; rowInd<result.getRows(); rowInd++)
		{
			for(int colInd=0; colInd<result.getCols(); colInd++)
			{
				result.set(rowInd, rowInd, alpha*result.get(rowInd, rowInd));
			}
		}
		return result;
	}
	
	public float getSum()
	{
		return sum;
	}
	
	/* dont use-pointer issues!
	protected FDMatrix csrmv(boolean transA, float alpha, SparseFMatrix sMatA, 
			FDMatrix vecX, float beta, FDMatrix vecY)
	{
		sMatA.sendToGPU();
		vecX.sendToGPU();
		vecY.sendToGPU();
		
		sMatA=sMatA.clone();


		int transAInt;
		if(transA)
		{
			transAInt=1;
		}
		else
		{
			transAInt=0;
		}
		cusparseScsrmv(handle, transAInt, sMatA.getRows(), sMatA.getCols(), sMatA.getNNZ(), 
				Pointer.to(new float[]{alpha}), descra, sMatA.valuesPointer, sMatA.rowsIndicesPointer,
				sMatA.colsIndicesPointer, vecX.gpuPointer, Pointer.to(new float[]{beta}), 
				vecY.gpuPointer);
		
		//sMatA.valuesPointer=null;
		//sMatA.rowsIndicesPointer=null;
		//sMatA.colsIndicesPointer=null;
		//sMatA.inGPU=false;
		
		return vecY;
	}
	*/
	
	/*
	protected SpraseFMatrix gemm(int transA, int transB, SparseFMatrix matA, SparseFMatrix matB, 
			SparseFMatrix matC) 
	{
		cusparseScsrgemm(handle, transA, transB, matA.getRows(), matB.getCols(), matA.getCols(),
				descra, matA.getNNZ(), matA.valuesPointer, matA.rowsIndicesPointer,
				matA.colsIndicesPointer, descra, int nnzB, jcuda.Pointer csrValB, jcuda.Pointer csrRowPtrB, jcuda.Pointer csrColIndB, cusparseMatDescr descrC, jcuda.Pointer csrValC, jcuda.Pointer csrRowPtrC, jcuda.Pointer csrColIndC) ;
	}
	*/
	
	protected FMatrix csrmm(int transA, int transB, float alpha, SparseFMatrix matA, FMatrix matB,
			float beta, FMatrix matC)
	{
		matA.sendToGPU();
		matB.sendToGPU();
		matC.sendToGPU();
		
		cusparseScsrmm2(matA.handle, transA, transB, matA.getRows(), matC.getCols(), matA.getCols(), 
				matA.getNNZ(), Pointer.to(new float[]{alpha}), matA.descra, matA.valuesPointer, 
				matA.rowsIndicesPointer, matA.colsIndicesPointer, matB.gpuPointer, matB.getRows(), 
				Pointer.to(new float[]{beta}), matC.gpuPointer, matC.getRows());
		
		return matC;
	}
	
	@Override
	public boolean equals(Object other)
	{
		getFromGPU();
		((SparseFMatrix)other).getFromGPU();
		if(nnz!=((SparseFMatrix)other).getNNZ())
		{
			return false;
		}
		else
		{
			for(int nzInd=0; nzInd<nnz; nzInd++)
			{
				if(values[nzInd]!=((SparseFMatrix)other).values[nzInd]
						|| rowIndices[nzInd]!=((SparseFMatrix)other).rowIndices[nzInd]
						|| colIndices[nzInd]!=((SparseFMatrix)other).colIndices[nzInd])
				{
					return false;
				}
			}
		}
		return true;
	}
	
	@Override
	public String toString()
	{
		if(inGPU)
		{
			Pointer cooRowsIndicesPointer=GPUMemoryManager.alloc(nnz*Sizeof.INT);
			cusparseXcsr2coo(handle, rowsIndicesPointer, nnz, getRows(),
					cooRowsIndicesPointer, CUSPARSE_INDEX_BASE_ZERO);
			
			values=new float[nnz];
			rowIndices=new int[nnz];
			colIndices=new int[nnz];
			
			cublasGetVector(nnz, Sizeof.FLOAT, valuesPointer, 1, Pointer.to(values), 1);
			cublasGetVector(nnz, Sizeof.INT, cooRowsIndicesPointer, 1, Pointer.to(rowIndices), 1);
			cublasGetVector(nnz, Sizeof.INT, colsIndicesPointer, 1, Pointer.to(colIndices), 1);
			
			GPUMemoryManager.free(cooRowsIndicesPointer);
		}
		return "Values: "+Arrays.toString(values)+"\n"
			  +"Rows  : "+Arrays.toString(rowIndices)+"\n"
			  +"Column: "+Arrays.toString(colIndices)+"\n";
	}
	
	private void writeObject(ObjectOutputStream oos) throws IOException 
    {
    	getFromGPU();
	    oos.defaultWriteObject();
	    oos.writeObject(rows);
	    oos.writeObject(columns);
	    oos.writeObject(values);
	    oos.writeObject(rowIndices);
	    oos.writeObject(colIndices);
	    oos.writeObject(inGPU);
	    oos.writeObject(nnz);
	    oos.writeObject(sum);
    }

	private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException 
	{
	    ois.defaultReadObject();
	    rows=(int)ois.readObject();
	    columns=(int)ois.readObject();
	    values=(float[])ois.readObject();
	    rowIndices=(int[])ois.readObject();
	    colIndices=(int[])ois.readObject();
	    inGPU=(boolean)ois.readObject();
	    nnz=(int)ois.readObject();
	    sum=(float)ois.readObject();
	}
	
}
