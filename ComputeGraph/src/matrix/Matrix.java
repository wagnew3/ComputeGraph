package matrix;

import java.io.Serializable;

public abstract class Matrix implements Serializable
{
	
	public static String workspaceDir="/home/willie/workspace/";
	
	public Matrix()
	{
		
	}
	
	public Matrix(double[] data) 
	{
		
	}
	
	public Matrix(double[][] data) 
	{
		
	}
	
	public Matrix(float[] data) 
	{
		
	}
	
	public Matrix(float[][] data) 
	{
		
	}
	
	public abstract int getRows();
	
	public abstract int getCols();
	
	public abstract int getLen();
	
	public abstract float get(int row, int col);
	
	public abstract void set(int row, int col, float val);
	
	public abstract float[] getData();
	
	public abstract Matrix getSubVector(int offset, int length);
	
	public abstract Matrix otrans();
	
	public abstract Matrix append(Matrix toAppend);
	
	public abstract Matrix mmult(Matrix toMultiplyBy);
	
	public abstract Matrix ommult(Matrix toMultiplyBy);
	
	public abstract Matrix oebemult(Matrix multVec);
	
	public abstract Matrix mscal(float toScaleBy);
	
	public abstract Matrix omscal(float toScaleBy);
	
	public abstract Matrix mad(Matrix toAddTo);
	
	public abstract Matrix omad(Matrix toAddTo);
	
	public abstract Matrix madScale(Matrix toAddTo, float scaleAddBy);
	
	public abstract Matrix omadScale(Matrix toAddTo, float scaleAddBy);
	
	public abstract Matrix msub(Matrix toSubtractBy, Matrix result);
	
	public abstract Matrix omsub(Matrix toSubtractBy);
	
	public abstract Matrix msubScale(Matrix toSubtractBy, float scaleSubBy);
	
	public abstract Matrix omsubScale(Matrix toSubtractBy, float scaleSubBy);
	
	public abstract float dot(Matrix toDotWith);
	
	public abstract Matrix matVecMultScale(Matrix mat, Matrix vec, float scaleSubBy);
	
	public abstract Matrix omatVecMultScale(Matrix mat, Matrix vec, float scaleSubBy);
	
	public abstract Matrix matVecMultScaleAdd(Matrix mat, Matrix vecMult, float scaleSubBy, Matrix vecAdd);
	
	public abstract Matrix omatVecMultScaleAdd(Matrix mat, Matrix vec, float scaleSubBy, Matrix vecAdd, Matrix result, float scaleResultBy);
	
	public abstract Matrix outProd(Matrix vecA, Matrix vecB, Matrix result);
	
	public abstract Matrix sgemm(boolean transposeA, boolean transposeB, Matrix toMultiplyBy, float alpha, float beta, Matrix toAdd, Matrix result);

	public abstract void clear();

	public abstract Matrix matVecMultScaleAddScale(Matrix mat, Matrix vecMult, float scaleSubBy, Matrix vecAdd,
			float scaleAddBy);

	public abstract Matrix scal(float alpha, int incx, Matrix result);

	public Matrix copyTo(Matrix mat) {
		// TODO Auto-generated method stub
		return null;
	}

	public Matrix sgemv(boolean b, float f, Matrix paramMatrix, Matrix matrix, int i, float g, Matrix objectiveDiff,
			int j, boolean c, Matrix objectiveDiff2) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
