import java.io.*;
import java.math.*;

public class DuckHunt {

	public static void main(String[] args) {
		hmm3();
	}
	
	/**
	 * Solution to HMM1. In HMM1 you should calculate
	 * the emission matrix for t=2 using the following 
	 * operations:
	 * 
	 * t=1		P(S_1)=π
	 * t=2		P(S_2|T,S_1)=πT
	 * t=2		P(S_2|T,S_1)E=P(O_2|S_2)
	 */
	private static void hmm1() {
		try {
			BufferedReader buf = new BufferedReader(new InputStreamReader(System.in));
			Matrix t = new Matrix(buf.readLine()); // transmission matrix
			Matrix e = new Matrix(buf.readLine()); // emission matrix
			Matrix s = new Matrix(buf.readLine()); // state probability distribution
		
			System.out.println(Matrix.multiply(Matrix.multiply(s, t), e).toString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Solution to HMM2. In HMM2 you should calculate the probability of a
	 * given observation sequence using the Forward algorithm.
	 */
	private static void hmm2() {
		try {
			BufferedReader buf = new BufferedReader(new InputStreamReader(System.in));
			Matrix t = new Matrix(buf.readLine()); // transmission matrix
			Matrix e = new Matrix(buf.readLine()); // emission matrix
			Matrix s = new Matrix(buf.readLine()); // state probability distribution
			int[] o = getObservations(buf.readLine());
			//Matrix fwm = forward(o);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	private static void hmm3() {
		try {
			BufferedReader buf = new BufferedReader(new InputStreamReader(System.in));
			Matrix t = new Matrix(buf.readLine()); // transmission matrix
			Matrix e = new Matrix(buf.readLine()); // emission matrix
			Matrix s = new Matrix(buf.readLine()); // state probability distribution
			Matrix fwm = forward(getObservations(buf.readLine()), t, e, s);
			int[] i = fwm.getAllMaxRowIndices();
			StringBuilder sb = new StringBuilder();
			for (int c=0; c<i.length; c++) {
				if (c > 0) sb.append(" ");
				sb.append(i[c]);
			}
			System.out.println(sb.toString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * The forward algorithm. Returns a matrix of size s*t
	 * (s rows and t columns) where s is the number of states and
	 * t is the number of time steps. The item on position a,b in
	 * the matrix corresponds to the probability of state A at time
	 * B.
	 */
	private static Matrix forward(int[] o, Matrix a, Matrix b, Matrix pi) {
		final int T = o.length;
		double[][] fwd = new double[a.rows()][T]; // a.rows = number of states
		for (int i = 0; i < a.rows(); i++) {
			// Init probabilities foreach state at time = 0
			fwd[i][0] = pi[i] * b[i][o[0]];
		}
		for (int t = 0; t < T-1; t++) {// foreach time step (or column)
			for (int j = 0; j < a.rows(); i++) { // foreach state (or row)
				fwd[j][t+1] = 0;
				for (int i = 0; i < a.rows(); i++) {
					fwd[j][t+1] += (fwd[i][t] * a[i][j]);
				}
				fwd[j][t+1] *= b[j][o[t+1]];
			}
		}
		return new Matrix(fwd);
	}
	
	private static int[] getObservations(String data) {
		String[] items = data.split(" ");
		int[] o = new int[items.length - 1];
		for(int i = 1; i < items.length; i++)
			v[i-1] = Integer.parseInt(items[i]);
		return o;
	}
	
	private static class Matrix {
		private double[][] matrix;
		
		/**
		 * Create a matrix from a string of data
		 * For example:
		 * 2 3 1 1 1 2 2 2
		 * Will be parsed as
		 * 1 1 1
		 * 2 2 2
		 */
		public Matrix(String data) {
			String[] items = data.split(" ");
			int rows = Integer.parseInt(items[0]);
			int columns = Integer.parseInt(items[1]);
			this.matrix = new double[rows][columns];
			int i = 2;
			for (int row = 0; row < rows; row++)
				for (int column = 0; column < columns; column++) {
					matrix[row][column] = Double.parseDouble(items[i]);
					i++;
				}
			
		}
		
		private Matrix(double[][] matrix) {
			this.matrix = matrix;
		}
		
		public static double round(double value, int places) {
			if (places < 0) throw new IllegalArgumentException();

			BigDecimal bd = new BigDecimal(value);
			bd = bd.setScale(places, RoundingMode.HALF_UP);
			return bd.doubleValue();
		}
		
		/**
		 * Perform the operation m1 * m2 and return the result.
		 */
		public static Matrix multiply(Matrix m1, Matrix m2) {
			double[][] a = m1.getData();
			double[][] b = m2.getData();
			int aRows = a.length;
	        int aColumns = a[0].length;
	        int bColumns = b[0].length;
	        double[][] result = new double[aRows][bColumns];

	        for (int i = 0; i < aRows; i++)
	            for (int j = 0; j < bColumns; j++)
	                for (int k = 0; k < aColumns; k++) {
						// Round off double to avoid floating point precision error
	                    result[i][j] = round(result[i][j] + a[i][k] * b[k][j], 10);
					}

	        return new Matrix(result);
		}
		
		/**
		 * Get the row index for the maximum element in the column
		 * whose index is given as parameter.
		 */ 
		public int getMaxRowIndex(int column) {
			int max_row = Integer.MIN_VALUE;
			for (int row = 0; row < matrix.length; row++) {
				if (matrix[row][column] > max_row) max_row = row;
			}
			return max_row;
		}
		
		/**
		 * Get all max row indices for this matrix.
		 */ 
		public int[] getAllMaxRowIndices() {
			int[] indices = new int[matrix[0].length];
			for (int column = 0; i < matrix[0].length; column++) {
				indices[column] = getMaxRowIndex(column);
			}
		}
		
		public int rows() {
			return this.matrix.length;
		}
		
		/**
		 * Return the string representation of this matrix.
		 * For example, if the matrix looks like this:
		 * 1 1 1
		 * 2 2 2
		 * The following string will be returned
		 * 2 3 1 1 1 2 2 2
		 * Where the first two numbers are the number of rows and columns
		 * respectively in that order.
		 */
		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(matrix.length + " " + matrix[0].length);
			for (int i = 0; i < matrix.length; i++)
				for (int j = 0; j < matrix[0].length; j++)
					sb.append(" " + matrix[i][j]);
			
			return sb.toString();
		}
		
		private double[][] getData() {
			return this.matrix;
		}
	}
}
