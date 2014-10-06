public class DuckHunt {

	public static void main(String[] args) {
		//hmm1(args);
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
	private static void hmm1(String[] args) {
		Matrix t = new Matrix(args[0]); // transmission matrix
		Matrix e = new Matrix(args[1]); // emission matrix
		Matrix s = new Matrix(args[2]); // state probability distribution
		
		System.out.println(Matrix.multiply(Matrix.multiply(s, t), e).toString());
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
	                for (int k = 0; k < aColumns; k++)
	                    result[i][j] += a[i][k] * b[k][j];

	        return new Matrix(result);
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
			sb.append(matrix.length + ' ' + matrix[0].length);
			for (int i = 0; i < matrix.length; i++)
				for (int j = 0; j < matrix[0].length; j++)
					sb.append(' ' + matrix[i][j]);
			
			return sb.toString();
		}
		
		private double[][] getData() {
			return this.matrix;
		}
	}
}
