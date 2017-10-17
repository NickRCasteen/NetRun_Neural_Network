

public class NetRun
{
	//DATA
	private static double Learning_Rate;
	private static int SoMB;
	private static int NoE;
	private static int mode;

	//TRAINING AND EXPECTED SET
	private static double[][] TrainSet;
	private static double[][] ExpectSet;
	
	//WEIGHTS
	private static double[][] wl1;
	private static double[][] wl2;

	//BIASES
	private static double[] bl1;
	private static double[] bl2;

	//ACTIVATIONS
	private static double[] al1;
	private static double[] al2;

	//GRADIENT BIAS
	private static double[][] gbl1;
	private static double[][] gbl2; //2nd number is size of minibatch

	//GRADIENT WEIGHTS
	private static double[][][] gwl1; //3rd number is size of minibatch
	private static double[][][] gwl2;

	//MISC TOOLS
	private static int[][] RESULTS;

	public NetRun(double[][] Data, double[][] pw1, double[][] pw2, double[] pb1, double[] pb2, double pLR, int pSMB, int pNE, int pm)
	{
		TrainSet = Data; //Load all of our massive list of parameters into the set.
		wl1 = pw1;
		wl2 = pw2;
		bl1 = pb1;
		bl2 = pb2;
		Learning_Rate = pLR;
		SoMB = pSMB;
		NoE = pNE;
		mode = pm;
		ExpectSet = new double[TrainSet.length][10]; //Make one of the same size.


		al1 = new double[bl1.length]; //Output same as bias. Size wise.
		al2 = new double[bl2.length];
		gbl1 = new double[bl1.length][SoMB];
		gbl2 = new double[bl2.length][SoMB];
		gwl1 = new double[bl1.length][TrainSet[0].length-1][SoMB]; //TrainSet[0].length returns # of columns, 785. Need -1 to make this 784.
		gwl2 = new double[bl2.length][bl1.length][SoMB]; //From hidden to out, so out's size is first. Minibatch size defines size of last bit. Why? Just wait

		RESULTS = new int[2][(int)al2.length]; //Index 0 is expected, index 1 is actual from output layer. Sum +1 for each output at [0] and +1@[1] for hits

		SetUp(); //Shuffle the rows of the data and read [0] for each row in the data to set up the expected array


		//ALL of the things are now initialized.
	}

	public void Run()
	{

		//THIS IS THE MAIN THRUST OF THE PROGRAM.
		//This will run through ALL rows, stopping every 10 rows, or however many rows make a minibatch, to update weights and biases.
		//This check for minibatch size is:
			//let n = index for current row
			//let Size_of_Minibatch = 10
			//Run through a row
			//if n+1 % Size_of_Minibatch, then update weights and biases.
			//Base case: if n is at 9, then because this check is AFTER doing a row, that means the next row is 10. Because we start at index 0,
				//completing 9 means we completed the 10th row and need to update. The first of the next minibatch, which modded with Size
				//of minibatch, will be 0.

		//Quick little fix:
		if(!(mode == 1))
		{
			NoE = 1; //If not mode 1, then we're demoing testing or demoing training, and need only do one epoch.
			
		}

		for(int i = 0; i < NoE; i++) //LOOP EPOCHS
		{
			InitializeResultsArray(); //Reset results array for a new epoch.
			SetUp(); //Re-Shuffle and assemble ExpectSet

			for(int n = 0; n < TrainSet.length; n++) //EVERY ROW. ie every input vector. Each input vector must be passed to all nodes via their weights
			{
				for(int k = 0; k < al1.length; k++) //Every node IN HIDDEN LAYER
				{

					//z = E XiWi + b --> 1/1+exp(-z)
					double summate = 0.0; //Resets for every hidden node
					for(int j = 1; j < TrainSet[0].length; j++) //Every input FOR THIS ROW. Starts at 1 to skip column for expected value.
					{

						double inpoot = (TrainSet[n][j]/255.0);
						//Data values are 0 to 255. Normalized with /255 gives values between 0 and 1, which will be much easier to work
						//with while having a minimal loss of data.
						//To save on time, instead of normalizing the whole dataset beforehand, we just run 1 input node's normalization
						//right here. That at least ensures Data integrity and keeps the expected output untouched easier than with
						//more loops.
						summate += (inpoot * wl1[k][j-1]); //Weight must start at [k][0] proper, so j-1.
					}

					double z = -(summate + bl1[k]);
					double down = 1.0d + Math.exp(z); //Down used to be sure this math is properly captured without order of operation weirdness
					al1[k] = (1.0d/down); //Put that into output for this node.

				}

				for(int k = 0; k < al2.length; k++) //Every Node in OUTPUT LAYER
				{

					double summate = 0.0;
					for(int j = 0; j < al1.length; j++) //Every output from hidden layer
					{
						summate += (al1[j] * wl2[k][j]);
					}
					double z = -(summate + bl2[k]);
					double down = 1.0d + Math.exp(z);
					al2[k] = (1.0d/down); //Same format here, just working with al1 as input instead.

				}
				CheckNetworkHitOrMiss(n); //Before we worry about gradients and such, we see how we did for this row!

				//Now al1 and al2 are populated

/*
=====================================================================================================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ## ~ UPDATE GRADIENTS ~ ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
=====================================================================================================================================================================
*/

				for(int k = 0; k < gbl2.length && mode == 1; k++) //Going backwards. For every bias in output layer
				{


					gbl2[k][n % SoMB] = (al2[k] - ExpectSet[n][k]) * al2[k] * (1.0d - al2[k]);
					//n % SoMB. This is the biggest part. We need to store all the gradients a bias got for each row.
					//These gradients being stored together only comes into play at the end of a minibatch, so we'll
					//only have as many as there are rows in a minibatch.
					//So, as far as a single row is concerned, with updating biases and gradients and getting outputs,
					//these other rows are basically "parallel universes" that we can ignore.
					//Thus, it's okay to just replace values in this array. They'll all be replaced come time for 
					//bias and weight updating.
					//Index logic. n % SoMB because, if SoMB is 10, for instance, n%SoMB will loop through 0-9 inclusive.
					//As said before, n%SomB = 0 is the first of a new minibatch, and thus the start of a new set of
					//gradients.


					for(int f = 0; f < al1.length; f++) //Every output from hidden layer. akl-1 * bgjl.
					{

						gwl2[k][f][n % SoMB] = al1[f] * gbl2[k][n % SoMB]; //Same idea here, just with a 3D array.

					}
				}
				
				double x = 0.0; //will be used for summation
				for(int k = 0; k < gbl1.length && mode == 1; k++) //Going backwards. For every bias in intermediate layer
                                {

					for(int d = 0; d < al2.length; d++) //Every node in output, the layer ahead
					{

						x += wl2[d][k] * gbl2[d][n % SoMB]; //Need to summate these beforehand.

					}
					
					gbl1[k][n % SoMB] = x * al1[k] * (1.0d - al1[k]); //apply the x as it needs to be.

					x = 0.0;
                                        
                                        for(int f = 1; f < TrainSet[0].length; f++)
                                        {

						double inpoot = (TrainSet[n][f]/255.0); //As we're using the Data again, we re-normalize.
						gwl1[k][f-1][n % SoMB] = inpoot * gbl1[k][n % SoMB]; //Apply to gradient for weights from in to hidden

                                        }
                                }

				
				

				//MINIBATCH END SIGNAL
				if(( n + 1 ) % SoMB == 0 && mode == 1) //Skip not in training mode. If NEXT row gets a zero when modded with SoMB...
				{
					//That means we just completed the last node for this minibatch!
					//End of minibatch. Perform weight/bias update
					
					//LAYER ONE
					for(int o = 0; o < al1.length; o++) //For every node in hidden layer.
					{
						
						double soomate = 0.0;
						for(int w = 0; w < SoMB; w++) //2 is going to equal Size of Minibatches
						{
							soomate += gbl1[o][w]; //We'll add up the gradients for this bias. Iterate through parallel universes!
						}
						bl1[o] = bl1[o] - (Learning_Rate / SoMB) * soomate; //And NOW it'll be okay.
						

						for(int f = 0; f < wl1[0].length; f++) //all the weights for this node.
						{
							soomate = 0.0;
							for(int t = 0; t < SoMB; t++) //again, 2 will be size of minibatch
							{
								soomate += gwl1[o][f][t]; //For this ONE weight, iterate each parallel universe.
							}
							wl1[o][f] = wl1[o][f] - (Learning_Rate / SoMB) * soomate;
						}
					}

					//LAYER TWO
                                        for(int o = 0; o < al2.length; o++) //For every node in output layer.
                                        {
                                                double soomate = 0.0;
                                                for(int w = 0; w < SoMB; w++) //2 is going to equal Size of Minibatches
                                                {
                                                        soomate += gbl2[o][w]; //We'll add up the gradients for this bias. Iterate through parallel universes!
                                                }
                                                bl2[o] = bl2[o] - (Learning_Rate / SoMB) * soomate; //And NOW it'll be-...waaait a second, did I copy/paste this???


                                                for(int f = 0; f < wl2[0].length; f++) //all the weights for this node.
                                                {
                                                        soomate = 0.0;
                                                        for(int t = 0; t < SoMB; t++) //again, 2 will be size of minibatch
                                                        {
                                                                soomate += gwl2[o][f][t]; //For this ONE weight, iterate each parallel universe.
                                                        }
                                                        wl2[o][f] = wl2[o][f] - (Learning_Rate / SoMB) * soomate; //I didn't copy THIS comment! It's new!
                                                }
                                        }

					
					// END OF MINIBATCH END SIGNAL
				}
				//END OF ROW
			}
			//print results
			PrintAccuracyTable();
		}


	}

//++++++++++++++++ SET UP ++++++++++++++++++++
	public void SetUp()
	{
		CSV_import shuffler = new CSV_import();
		TrainSet = shuffler.shuffleData(TrainSet); //shuffle up

		for(int row1 = 0; row1 < ExpectSet.length; row1++) //For all the many, many rows in expect
		{
			for(int col2 = 0; col2 < ExpectSet[0].length; col2++) //And all the many, but not AS many columns
			{
				if((int) TrainSet[row1][0] == col2)
				{
					//If there's, say, 10 columns per row, because there's 10 expected, we want a single 1 in one of those boxes.
					//The inputs are 0 to 9 (which we can also cheekily use as indexes for this size 10 array).
					//We know TrainSet's first column is full of all the expected values for their respective rows, so if col2, as it grows,
					//comes to equal what's in the 0th column, then we know exactly what the expected output is AND where to put the corresponding
					//1. Math is so fun.
					ExpectSet[row1][col2] = 1.0; //Since 0th entry for each row is 0-9, we can use this as an index to put our output 1.
					RESULTS[0][col2] += 1; //adds one to that slot in the results array
				}
				else
				{
					ExpectSet[row1][col2] = 0.0; //Makes all 0
				}
			}
			
		}

		shuffler = null;
		
	}


	private void InitializeResultsArray()
	{

		for(int q = 0; q < 2; q++)
		{
			for(int w = 0; w < RESULTS[0].length; w++)
			{
				RESULTS[q][w] = 0; //Make every slot 0.
			}
		}

	}


//++++++++++++++++ TOOLS ++++++++++++++++++++
	private void CheckNetworkHitOrMiss(int row) //Will be able to see output. No params. This will set the RESULT array.
	{
		//Runs after every node in Input into node

		//The expected out array will reveal what column to set in the RESULTS array, and the output we examine is the row.
		//For every out and expect output, of which there are 10 slots
		double storednum = 0.0;
		int storedindex = -1;

		//Okay, so we're going to go down the entirety of output_out[]. i < OutSize. If output_out[i] > num then num = output_out[i] and storedindex = i.
		//After we get the index containing the biggest number, if expected_output_y[storedindex] == 1, it's a hit.

		for(int i = 0; i < al2.length; i++)
		{
			if(al2[i] > storednum)
			{
				storednum = al2[i];
				storedindex = i;
			}
		}
		//What follows is a catch for a fatal error in learning. And I mean a BAD error.
		if(storedindex == -1)
		{
			System.out.println("CHECK HIT OR MISS MASSIVE ERROR CATCH");
			System.out.println("STORED INDEX NEVER UPDATED. ALL NUMBERS IN OUTPUT ARE SOMEHOW 0");
			System.out.println("COULD THEY BE NEGATIVE? PRINTING OUTPUT VALUES.");
			for(int j = 0; j < al2.length; j++)
			{
				System.out.printf(" " + al2[j] + " ");
			}
			System.out.println("");
			System.out.println("ABORTING PROGRAM. SET UP SOME NUMBERS PRINTS AND FIX THIS.");
			System.exit(0);
		}

		if(ExpectSet[row][storedindex] == 1.0)
		{
			//It's a hit! ++ in RESULT[1][???]
			double info = TrainSet[row][0]; //The number still exists inside TrainSet, may as well use it. Does NOT do the 0/1 thing, keep index 0 as
							//It's the expected, ya goose.
			RESULTS[1][(int) info] += 1;
		}
		//If Cost > 0.1, keep it. No plus. You fail. So that's one row done and checked. The RESULTS array will be printed at the end of the epoch.

	}


	private void PrintAccuracyTable() //Just needs RESULTS.
	{
		//Runs after every epoch.
		//As per Mike's specifications.
		//0 = actual / full
		//Use RESULTS table. The ++s I've had should have thousands in each slot.
		int SumExpect = 0;
		int SumActual = 0;
		double Percentage = 0.0;

		for(int i = 0; i < al2.length; i++)
		{
			System.out.printf(i + " = " + RESULTS[1][i] + "/" + RESULTS[0][i] + "  "); //Print each one in specified format.
			if(i == 5)
			{
				System.out.println(" "); //New line after 5
			}
			SumExpect = SumExpect + RESULTS[0][i]; //Just add while we're here. One loop, please.
			SumActual = SumActual + RESULTS[1][i]; //Ditto.
		}
		
		Percentage = (double) SumActual / (double) SumExpect; //Gotta typecast to get our double percentage.

		Percentage *= 100.0;

		System.out.println("Accuracy = " + SumActual + "/" + SumExpect + " = " + Percentage + "%"); //The full accuracy printed.

		System.out.println("");

	}

	
	//This is a debug function, waits for user to press space. Made pausing the program easy.
	public void cont()
	{
		System.out.println("Press Enter key to continue...");
		try
        	{
            		System.in.read();
        	}  
        	catch(Exception e)
        	{
		}
	}


//++++++++++++++++ GETTERS ++++++++++++++++++++
//These are for system, so it can save the biases and do all the file writing whatsit.
	public double[] GetBiasHid()
	{
		return bl1;
	}

	public double[] GetBiasOut()
	{
		return bl2;
	}

	public double[][] GetWeightsIn2Hid()
	{
		return wl1;
	}

	public double[][] GetWeightsHid2Out()
	{
		return wl2;
	}
}
