import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

class Player {
	/*********************************************************************
	 ********************************************************************* 
	 * 						Player skeleton
	 *********************************************************************
	 *********************************************************************/
	private final double HMM_CORRECTNESS = 0;
	private final double EMISSION_PROBABILITY = 0.7;
	private final int SEQUENCE_LENGHT = 23;
	private final Action DO_NOT_SHOOT = new Action(-1, -1);
	private static int hits;
	private static int shots;
	private Oracle oracle;
	private boolean isInitialized = false;
	private HMM[] storedHmm;
	
	public Player() {
		this.oracle = new Oracle();
	}
	
	public Action shoot(GameState state, Deadline due) {
		/* Make sure we have enough observations. */
		if (state.getBird(0).getSeqLength() < SEQUENCE_LENGHT) {
			return DO_NOT_SHOOT;
		}

		HMM[] hmm = new HMM[state.getNumBirds()];
		double[] hmmCorrectness = new double[state.getNumBirds()];
		for (int i = 0; i < hmmCorrectness.length; i++) {
			hmmCorrectness[i] = Double.MIN_VALUE;
		}

		/* Create an HMM for each bird. */
		for (int i = 0; i < state.getNumBirds(); i++) {
			hmm[i] = new HMM(getObservationSequence(state.getBird(i)));
		}

		/* Tune the models to reflect the underlying observation sequence. */
		for (int i = 0; i < state.getNumBirds(); i++) {
			if (state.getBird(i).isAlive()) {
				hmmCorrectness[i] = hmm[i].tune();
			}
		}
		
		/* Save the HMMs. */
		storedHmm = hmm;

		/* Pick a bird to shoot. In order to maximize score, the following
		 * aspects are taken into account:
		 * 1) Consider only the most correct model
		 * 2) Don't shoot if model is not correct enough
		 * 3) Don't shoot if flight path seems unpredictable 
		 * 4) Don't shoot if bird is already dead 
		 * 5) Don't shoot if the bird looks like a stork */
		int target = max(hmmCorrectness);
		int emission = 0; // One of the observation values in Constants.java
		if (state.getBird(target).isAlive() && 
			hmmCorrectness[target] >= HMM_CORRECTNESS &&
			!oracle.isBlackStork(getObservationSequence(state.getBird(target)))) {
			double[] emissionProbabilities = hmm[target].getEmissionProbabilities();
			double max = emissionProbabilities[0];
			for (int j = 1; j < emissionProbabilities.length; j++) {
				if (emissionProbabilities[j] > max) {
					max = emissionProbabilities[j];
					emission = j;
				}
			}
			if (max > EMISSION_PROBABILITY) {
				shots++;
				return new Action(target, emission);
			}
		}
		return DO_NOT_SHOOT;
	}
	
	/** Return the index for the maximum value in 'a'. */
	private int max(double[] a) {
		int index = 0;
		double max = a[0];
		for (int i = 1; i < a.length; i++) {
			if (a[i] > max) {
				max = a[i];
				index = i;
			}
		}
		return index;
	}

	/** Get an observation sequence for a bird. */
	private int[] getObservationSequence(Bird bird) {
		int[] observations = new int[bird.getSeqLength()];
		for (int i = 0; i < bird.getSeqLength(); i++) {
			observations[i] = bird.getObservation(i);
		}
		return observations;
	}

	public int[] guess(GameState state, Deadline due) {
		if (isInitialized) {
			// The oracle is initialized, make a guess!
			return oracle.guess(state);
		} else {
			// Guess all pigeons
			return oracle.getPigeonArray(state.getNumBirds());
		}
	}

	public void hit(GameState state, int bird, Deadline due) {
		hits++;
	}

	public void reveal(GameState state, int[] revealed, Deadline due) {
		if (!isInitialized) {
			oracle.initialize(state, revealed, storedHmm);
			isInitialized = true;
		}
	}

	public static int getShots() {
		return shots;
	}

	public static int getHits() {
		return hits;
	}
	
	/*********************************************************************
	 ********************************************************************* 
	 * 						  Guess species
	 *********************************************************************
	 *********************************************************************/
	private class Oracle {
		private int[] species = new int[] { 
				Constants.SPECIES_PIGEON, 
				Constants.SPECIES_RAVEN, 
				Constants.SPECIES_SKYLARK,
				Constants.SPECIES_SWALLOW,
				Constants.SPECIES_SNIPE,
				Constants.SPECIES_BLACK_STORK,
		};
		/* Will contain reference birds, in the order defined by our species array above. */
		private ReferenceBird[] referenceBirds = new ReferenceBird[Constants.COUNT_SPECIES];
		private final int DRILLING_LENGTH = 5;
		
		private class ReferenceBird {
			/* A drilling sequence unique for this species. */
			public int[] drillingSequence;
			/* Type of species. One of the species in Constants.java */
			public int species;
			/* The complete observation sequence for this bird. */
			public int[] observationSequence;
			
			public ReferenceBird(Bird bird, int species) {
				this.species = species;
				this.observationSequence = new int[bird.getSeqLength()];
				for (int i = 0; i < bird.getSeqLength(); i++) {
					observationSequence[i] = bird.getObservation(i);
				}
			}
		}
		
		public int[] getPigeonArray(int nBirds) {
			int[] pigeons = new int[nBirds];
			for (int i = 0; i < nBirds; i++) {
				pigeons[i] = Constants.SPECIES_PIGEON;
			}
			return pigeons;
		}
		
		public void initialize(GameState gState, int[] revealed, HMM[] hmm) {
			// Step 1: Get reference birds from our GameState
			for (int i = 0; i < Constants.COUNT_SPECIES; i++) {
				referenceBirds[i] = getReferenceBird(species[i], gState, revealed); // returns null if species[i] does not exist in gState
			}
			
			// Step 2: Decode the state sequences for each of the reference birds
			for (int i = 0; i < Constants.COUNT_SPECIES; i++) {
				// Step 3: Extract the drilling sequence
				if (referenceBirds[i] != null) {
					referenceBirds[i].drillingSequence = getDrillingSequence(referenceBirds[i], hmm[i].decode());
				}
			}
		}
		
		/** Return a reference bird of the type given as argument. The reference bird 
		 * chosen will be the one with with the longest observation sequence. */
		private ReferenceBird getReferenceBird(int type, GameState state, int[] revealed) {
			int max = -1;
			for (int i = 0; i < state.getNumBirds(); i++) {
				if (type == revealed[i] && 
					state.getBird(i).getSeqLength() > max) {
					max = i;
				}
			}
			if (max == -1) {
				// No such bird
				return null;
			}
			return new ReferenceBird(state.getBird(max), revealed[max]);
		}
		
		/** Get the drilling sequence of maximum length DRILLING_LENGHT for the reference bird given as 
		 * argument, e.g return the corresponding observation sequence that matches the longest 
		 * continuous state subsequence of 3:s. */
		public int[] getDrillingSequence(ReferenceBird bird, ArrayList<Integer> states) {
			int start = -1; // Inclusive
			int end = -2; // Exclusive
			int count = 1;
			boolean isCounting = false;
			for (int i = 0; i < states.size(); i++) {
				if (isCounting) {
					if (!states.get(i).equals(3)) {
						if (count > end - start + 1) {
							// New LCS
							start = i - count + 1;
							end = count;
						} else {
							isCounting = false;
							count = 1;
						}
					} else {
						count++;
						if (count == DRILLING_LENGTH) {
							// Trim the drilling length in order to make isSubsequence faster
							break;
						}
					}
				} else if (states.get(i).equals(3)) {
					isCounting = true;
				}
			}
			if (end == -2) {
				// No drills :(
				return new int[0];
			}
			return Arrays.copyOfRange(bird.observationSequence, start, end);
		}
		
		private int[] getObservationSequence(Bird bird) {
			int[] observations = new int[bird.getSeqLength()];
			for (int i = 0; i < bird.getSeqLength(); i++) {
				observations[i] = bird.getObservation(i);
			}
			return observations;
		}
		
		/** Try to guess each of the species in the game state given as argument. */
		public int[] guess(GameState gState) {
			int[] guess = new int[gState.getNumBirds()];
			Arrays.fill(guess, Constants.SPECIES_UNKNOWN);
			for (int i = 0; i < gState.getNumBirds(); i++) {
				/* If the drilling sequence for the species Y (known) is a part of the observation sequence 
				 * for X (unknown) then X is considered to be of the same species as Y. */
				for (int j = 0; j < Constants.COUNT_SPECIES; j++) {
					if (referenceBirds[j] != null &&
						referenceBirds[j].drillingSequence.length == DRILLING_LENGTH &&
						isSubsequence(referenceBirds[j].drillingSequence, getObservationSequence(gState.getBird(i)))) {
						// X is of type referenceBirds[j]!
						guess[i] = referenceBirds[j].species;
						break;
					}
				}
			}
			return guess;
		}
		
		private boolean isSubsequence(int[] drillingSequence, int[] observationSequence) {
			int i = Collections.indexOfSubList(Arrays.asList(observationSequence), 
				Arrays.asList(drillingSequence)); // source, target
			return i == -1 ? false : true;
		}

		public boolean isBlackStork(int[] observations) {
			return false;
		}
	}

	/*********************************************************************
	 ********************************************************************* 
	 * 						Hidden Markov Model
	 *********************************************************************
	 *********************************************************************/
	private class HMM {
		private int nStates, nEmissions;
		private final int STEPS = 80; // Maximum number of steps in Baum-Welch
		private int[] observations;
		private double[][] A, B;
		private double[] pi;

		private class InitValues {
			private double oneNine = (double) 1 / (double) 9;
			/* Transmission matrix, probability of transmitting from one state to another. */
			public double[][] A =  {
					// Migrating 	Circling 	Hunting 		Drilling	Zigzag
					{0.2,			0.2, 		0.2, 			0.2, 		0.2}, // Migrating
					{0.2, 			0.2, 		0.2, 			0.2, 		0.2}, // Circling
					{0.2, 			0.2, 		0.2, 			0.2, 		0.2}, // Hunting
					{0.2, 			0.2, 		0.2, 			0.2, 		0.2}, // Drilling
					{0.2, 			0.2, 		0.2, 			0.2, 		0.2}  // Zigzag 
			};
			/* Emission matrix, probability of a certain emission in each state. */
			public double[][] B = {
					// Up-left	Up 		Up-right 	Left 	Stopped		Right	Down-left	Down	Down-right
					{0.075, 	0.02, 	0.075, 		0.33, 	0.0, 		0.33, 	0.075, 		0.02, 	0.075}, 	// Migrating
					{oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine}, 	// Circling
					{oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine}, 	// Hunting
					{oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine, 	oneNine,oneNine},	// Drilling
					{0.125, 	0.125, 	0.125, 		0.125, 	0.0, 		0.125, 	0.125, 		0.125, 	0.125} 		// Zigzag
			};
			/* State probability distribution. */
			public double[] pi = {
					// Migrating	Circling	Hunting		Drilling	Zigzag
					0.2, 			0.2, 		0.2, 		0.2, 		0.2
			};
		}

		/* Creates an HMM with initial values and the observation sequence provided. */
		public HMM(int[] observations) {
			InitValues myMatrices = new InitValues();
			this.observations = observations;
			this.A = myMatrices.A;
			this.B = myMatrices.B;
			this.pi = myMatrices.pi;
			this.nStates = pi.length;
			this.nEmissions = B[0].length;
		}

		/** Tune A, B and pi using Baum-Welch. Returns the correctness of this model. */
		public double tune() {
			double oldLogProb = Double.NEGATIVE_INFINITY; // Do not change to Double.MIN_VALUE
			double logprob = baumWelch();
			int i = 0;
			while (i < STEPS && logprob > oldLogProb) {
				oldLogProb = logprob;
				logprob = baumWelch();
				i++;
			}
			return Math.exp(logprob);
		}

		/** One step of Baum-Welch algorithm. Implemented from 
		 * http://dd2380.csc.kth.se/uploads/HMM_Tutorial_Stamp.pdf */
		public double baumWelch() {
			final int T = observations.length;

			/***************** Alpha pass. ********************/
			double[][] alpha = new double[nStates][T];
			double[] c = new double[T];

			// Compute a[i][0]
			c[0] = 0;
			for (int i = 0; i < nStates; i++) {
				alpha[i][0] = pi[i] * B[i][observations[0]];
				c[0] += alpha[i][0];
			}

			// Scale the alpha[i][0]
			c[0] = div(1.0, c[0]);
			for (int i = 0; i < nStates; i++) {
				alpha[i][0] = c[0] * alpha[i][0];
			}

			// Compute alpha[i][t]
			for (int t = 1; t < T; t++) {
				c[t] = 0;
				for (int i = 0; i < nStates; i++) {
					alpha[i][t] = 0;
					for (int j = 0; j < nStates; j++) {
						alpha[i][t] = alpha[i][t] + alpha[j][t - 1] * A[j][i];
					}
					alpha[i][t] = alpha[i][t] * B[i][observations[t]];
					c[t] += alpha[i][t];
				}

				// Scale alpha[i][t]
				c[t] = div(1.0, c[t]);
				for (int i = 0; i < nStates; i++) {
					alpha[i][t] = c[t] * alpha[i][t];
				}
			}

			/***************** Beta pass. ********************/
			double[][] beta = new double[nStates][T];

			// Let beta[i][T-1]=1 scaled by c[T-1]
			for (int i = 0; i < nStates; i++) {
				beta[i][T - 1] = c[T - 1];
			}

			// Beta-pass
			for (int t = T - 2; t >= 0; t--) {
				for (int i = 0; i < nStates; i++) {
					beta[i][t] = 0;
					for (int j = 0; j < nStates; j++) {
						beta[i][t] = beta[i][t]  + (A[i][j] 
								* B[j][observations[t + 1]] 
										* beta[j][t + 1]);
					}
					// Scale the beta[i][t] with same scale factor as alpha[i][t]
					beta[i][t] = c[t] * beta[i][t];
				}
			}

			// Compute gamma[i][t] and digamma[i][j][t]
			double[][] gamma = new double[nStates][T]; 
			double[][][] digamma = new double[nStates][nStates][T]; 

			for (int t = 0; t < T - 1; t++) {
				double denom = 0;
				for (int i = 0; i < nStates; i++) {
					for (int j = 0; j < nStates; j++) {
						denom = denom + alpha[i][t] * A[i][j] * 
								B[j][observations[t + 1]] * beta[j][t + 1];
					}
				}
				for (int i = 0; i < nStates; i++) {
					gamma[i][t] = 0;
					for (int j = 0; j < nStates; j++) {
						digamma[i][j][t] = div((alpha[i][t] * A[i][j] * 
								B[j][observations[t + 1]] * beta[j][t + 1]), denom);
						gamma[i][t] += digamma[i][j][t];
					}
				}
			}

			/***************** Re-estimate A, B and pi. ********************/
			// Re-estimate pi
			for (int i = 0; i < nStates; i++) {
				pi[i] = gamma[i][0];
			}

			// Re-estimate A
			for (int i = 0; i < nStates; i++) {
				for (int j = 0; j < nStates; j++) {
					double numer = 0;
					double denom = 0;
					for (int t = 0; t < T - 1; t++) {
						numer = numer + digamma[i][j][t];
						denom = denom + gamma[i][t];
					}
					A[i][j] = div(numer, denom);
				}
			}

			// Re-estimate B
			for (int i = 0; i < nStates; i++) {
				for (int j = 0; j < nEmissions; j++) {
					double numer = 0;
					double denom = 0;
					for (int t = 0; t < T - 1; t++) {
						if (observations[t] == j) {
							numer += gamma[i][t];
						}
						denom = denom + gamma[i][t];
					}
					B[i][j] = div(numer, denom);
				}
			}

			/***************** Compute log(c[i]). ********************/
			double logprob = 0;
			for (int i = 0; i < T; i++) {
				logprob += Math.log(c[i]);
			}

			return -logprob;
		}

		private double div(double a, double b) {
			if (b == 0) {
				return 0;
			}
			return a / b;
		}

		private ArrayList<ArrayList<Integer>> getPathsMatrix() {
			ArrayList<ArrayList<Integer>> paths = new ArrayList<ArrayList<Integer>>(nStates);
			for (int i = 0; i < nStates; i++) {
				paths.add(new ArrayList<Integer>());
			}
			return paths;
		}
		
		public double[] getEmissionProbabilities() {
			ArrayList<Integer> path = decode();
			int cState = path.get(path.size() - 1);
			double[] emissionProbabilities = new double[Constants.COUNT_MOVE];
			for (int i = 0; i < emissionProbabilities.length; i++) {
				for (int state = 0; state < nStates; state++) {
					// A[foo] contains the probabilities of transmitting
					// from foo to each of the states in A.
					double[] row = A[state];
					for (int nextState = 0; nextState < row.length; nextState++) {
						emissionProbabilities[i] += (state == cState ? 1 : 0) * 
								A[state][nextState] * B[nextState][i];
					}
				}
			}
			return emissionProbabilities;
		}
		
		/** Viterbi decoding. Implemented from 
		 * http://dd2380.csc.kth.se/upload/Tutorial-HMM2-2014.pdf */
		public ArrayList<Integer> decode() {
			double[][] delta = new double[observations.length][nStates];
			ArrayList<ArrayList<Integer>> paths = getPathsMatrix();
			
			/***************** Initialize delta[i][t] ********************/ 
			for (int i = 0; i < nStates; i++) {
				delta[0][i] = Math.log(pi[i]) + Math.log(B[i][observations[0]]);
				paths.get(i).add(i);
			}

			/***************** Foreach t>0 ********************/ 
			for (int t = 1; t < observations.length; t++) {
				ArrayList<ArrayList<Integer>> nextPaths = getPathsMatrix();
				for (int cState = 0; cState < nStates; cState++) {
					double maxProbability = Double.NEGATIVE_INFINITY;
					int maxState = 0;
					for(int i = 0; i < nStates; i++) {
						double nextProbability = delta[t - 1][i] + Math.log(A[i][cState]) + 
								Math.log(B[cState][observations[t]]);
						if (nextProbability > maxProbability) {
							maxProbability = nextProbability;
							maxState = i;
						}
					}
					delta[t][cState] = maxProbability;
					nextPaths.get(cState).addAll(paths.get(maxState));
					nextPaths.get(cState).add(cState);
				}
				paths = nextPaths;
			}
			
			/***************** Probability of best path, find max(delta) ********************/ 
			double maxProbability = delta[observations.length - 1][0];
			int maxState = 0;
			for (int i = 1; i < nStates; i++) {
				double nextProbability = delta[observations.length - 1][i];
				if (nextProbability > maxProbability) {
					maxProbability = nextProbability;
					maxState = i;
				}
			}
			return paths.get(maxState);
		}
	}
}
