import java.util.Random;
import java.util.Scanner;
import java.io.*;
import java.util.Arrays;
import java.util.List;


public class system
{

	//Initialize all the arrays you will need

	//Constant learning rate
	private final double learning_rate = 3.0;

	//Constant size of minibatches. Just in case I wanna quickly change it.
	private final int size_of_minibatch = 10;



	//Number of epochs
	private final int number_of_epochs = 30;

	// SPECIAL VARIABLES
	//Size of Hidden Layer
	private static int HidSize = 100;

	//Size of OutputLayer
	private static int OutSize = 10; 

	//Size of Input Layer
	private static int InSize = 784; //Number of INPUT NODES.

	// SPECIAL VARIABLES



	//The biases for the hidden layer
	private static double[] bias_hidden = new double[HidSize];

	//The biases for the output layer. Yes, it's not standard neural network language but it helps me.
	private static double[] bias_out = new double[OutSize];

	//The weights going from input to the hidden layer. It's structured backwards, hidden_layer_size x input_layer_size, so the name is just for me
	private static double[][] weights_in_to_hidden = new double[HidSize][InSize];

	//The weights going from hidden to output layer. Again, the name is for me.
	private static double[][] weights_hidden_to_out = new double[OutSize][HidSize];

	//Hoo-wee, that's a LOT of arrays.

	public void start()
	{
		boolean trained = false; //Activates the options 3-6
		boolean going = true; //Keeps us in this menu
		boolean biasandweightsimported = false; //Lets the method below know whether or not we need to initialize out weights and biases.
		Scanner keyboard = new Scanner(System.in);
		while(going)
		{
			//Printout
			System.out.println("Type the number of the option you wish to perform: ");
			System.out.println("[1]: Import Weights and Biases From File");
			System.out.println("[2]: Run Training with current weights and biases (initially random)");
			if(trained)
			{
				System.out.println("[3]: Demo Training Set");
				System.out.println("[4]: Demo Testing Set");
				System.out.println("[5]: Export Weights and Biases to File");
				System.out.println("[6]: Clear Weights and Biases.");
			}
			System.out.println("[7]: Exit Program");

			int myint = keyboard.nextInt(); //Listens for input. Probably error-prone, but it functions.

			if(myint == 1) //Import. Very IMPORTant.
			{
				LoadModelFromFile();
				biasandweightsimported = true; //As we imported, we obviously set this to true.
				trained = true; //If the network is pre-trained, well we might as well be able to see everything.
			}
			else if(myint == 2) //Train with current biases.
			{
				System.out.println("Running Training");
				this.BeginNetwork(1,biasandweightsimported);
				trained = true;
				biasandweightsimported = true; //I set this to true just in case we wanna keep running with the same biases to edge a few extra points
			}
			else if(myint == 3 && trained) //Run through unshuffled training set once.
			{
				System.out.println("Demoing Training");
				this.BeginNetwork(2,biasandweightsimported); //Pass in mode 2 to run training but perform no operations for gradients or shuffling
			}
			else if(myint == 4 && trained) //Run through testing.
			{
				System.out.println("Demoing Testing");
				this.BeginNetwork(3,biasandweightsimported); //Ditto, but with BRAND NEW DATA
			}
			else if(myint == 5 && trained) //Export to file
			{
				SaveModelToFile();
			}
			else if(myint == 6 && trained) //Just sets two values to false so that they system will re-initialize the bias and weights. Cheeky!
			{
				//Clear weights and biases
				trained = false;
				biasandweightsimported = false;
			}
			else if(myint == 7) //exunt
			{
				going = false;
			}
			else //How dare
			{
				System.out.println("Invalid Input.");
			}
		}
	
	}

	public void BeginNetwork(int modetopass, boolean impwandbs)
	{
		boolean one; //one and two are just parameters to pass into the csv_importer when we eventually get to that.
		boolean two;
		if(modetopass == 1) //Training Set AND Training
		{
			one = true;
			two = true;
			initializer initi = new initializer();
			if(!(impwandbs)) //If we did NOT import weights and biases (Or haven't run before), we must initialize ourself.
			{
				weights_in_to_hidden = initi.InitializeWeights(weights_in_to_hidden, InSize, HidSize);
				weights_hidden_to_out = initi.InitializeWeights(weights_hidden_to_out, HidSize, OutSize);
				bias_hidden = initi.InitializeBias(bias_hidden);
				bias_out = initi.InitializeBias(bias_out);
			}
			initi = null; //Cleaning
			
		}
		else if(modetopass == 2) //Training Set, NOT Training
		{
			one = true;
			two = false;
		}
		else //Testing Set Demo
		{
			one = false;
			two = false;
		}

		CSV_import porter = new CSV_import(); //imports based on parameters

		NetRun ai = new NetRun(porter.csvimport(one, two), weights_in_to_hidden, weights_hidden_to_out, bias_hidden, bias_out, learning_rate, size_of_minibatch, number_of_epochs, modetopass); //runs and prints. Loooots of parameters.

		porter = null; //Free up memory.

		ai.Run();

		//The next 4 statements takes ai's weights and biases as our own. Of course, if we're not in mode 1, what we pass in is what we'll get back.
		bias_hidden = ai.GetBiasHid(); 
		bias_out = ai.GetBiasOut();
		weights_in_to_hidden = ai.GetWeightsIn2Hid();
		weights_hidden_to_out = ai.GetWeightsHid2Out();

		ai = null; //This is to destroy the class and its data for memory sake.
	

	}


	public void LoadModelFromFile()
	{
		//This code is largely adapted from the code used to read csv files in the CSV_import class
		//Credits for this code there!
	
		String csvFile = System.getProperty("user.dir") + "/SavedModel.csv"; //This includes the name of the file we save to.

		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";

		try
		{
			br = new BufferedReader(new FileReader(csvFile));
			int realindex = 0; //This is needed because when we're putting data into arrays, they need to start at 0.
			int rowindex = 0;
			int BiasHidendex = 0;
			int BiasOutendex = BiasHidendex + 1;
			int WIn2Hidendex = BiasOutendex + weights_in_to_hidden.length;
			int WHid2Outendex = WIn2Hidendex + weights_hidden_to_out.length;

			//So, the data is set up in a particular way. Hidden bias first, output bias second, weights in to hidden next and finally weights hid to out
			//These "endex"s store at what row each array's data ends. The bias for hidden will always be one, and biases after will also be +1 from the
			//previous bias. After that, the ends for the weights come their own legnth after the previous' endex.

			//This allows the size of the weights exported to the arbitrary. Just make sure the sizes between import and export match!
		
			while ((line = br.readLine()) != null)
			{
				int columnindex = 0;
				String[] row = line.split(cvsSplitBy);

				for(String rowdata : row)
				{
					if(rowindex == BiasHidendex) //Biases occupy a single row, a straightshot of data.
					{
						bias_hidden[columnindex] = Double.parseDouble(rowdata);
					}
					else if(rowindex == BiasOutendex)
					{
						bias_out[columnindex] = Double.parseDouble(rowdata);
					}
					else if(rowindex > BiasOutendex && rowindex <= WIn2Hidendex) //Meanwhile, weights have their full shape represented.
					{
						weights_in_to_hidden[realindex][columnindex] = Double.parseDouble(rowdata); //rowindex must be in this range
					}
					else if(rowindex > WIn2Hidendex && rowindex <= WHid2Outendex)
					{
						weights_hidden_to_out[realindex][columnindex] = Double.parseDouble(rowdata); //Extra note: endex is inclusive
					}
					columnindex++;
				}

				//We'll ++ to realindex for all times rowindex is above BiasOutendex. By the time we get here, we've done the 0th column.
				//So it'll add add add and then reset when rowindex passes a checkpoint, here defined by our endexes
				if(rowindex > BiasOutendex)
				{
					realindex++;
				}
				rowindex++;
				//If the row index before equaled an endex, reset realindex. We're in new territory.
				if(rowindex-1 == WIn2Hidendex || rowindex-1 == WHid2Outendex)
				{
					realindex = 0;
				}

				//So, recap: endexes determine what array we're setting AND passing it leads to resetting the realindex for the next array.
			}
		}
		catch(IOException e)
		{
		}

	}


	public void SaveModelToFile()
	{
		//This method of writing to file is adapted from Shaw from Stack Overflow!
		//https://stackoverflow.com/questions/30073980/java-writing-strings-to-a-csv-file
		String csvFile = System.getProperty("user.dir") + "/SavedModel.csv";
		
		try
		{
			FileWriter writer = new FileWriter(csvFile);

			//HIDDEN BIAS WRITEOUT
			for(int bh = 0; bh < bias_hidden.length; bh++) //Hidden biases on one row to save space
			{
				writer.append(Double.toString(bias_hidden[bh]));
				if(!(bh+1 > bias_hidden.length)) //gotta have a comma for \n I think
				{
					writer.append(",");
				}
			}

			writer.append("\n");
			
			//OUTPUT BIAS WRITEOUT
			for(int bo = 0; bo < bias_out.length; bo++) //Output biases on one row to save space
			{
				writer.append(Double.toString(bias_out[bo]));
				if(!(bo+1 > bias_out.length)) //gotta have a comma for \n I think
				{
					writer.append(",");
				}
			}

			writer.append("\n");

			//WEIGHTS IN 2 HIDDEN WRITEOUT
			for(int wi2hr = 0; wi2hr < weights_in_to_hidden.length; wi2hr++) //every row
			{
				for(int wi2hc = 0; wi2hc < weights_in_to_hidden[0].length; wi2hc++) //every column
				{
					writer.append(Double.toString(weights_in_to_hidden[wi2hr][wi2hc]));
					if(!(wi2hc+1 > weights_in_to_hidden[0].length))
					{
						writer.append(",");
					}
				}
				writer.append("\n");
			}


			//WEIGHTS HIDDEN 2 OUT WRITEOUT
			for(int wi2hr = 0; wi2hr < weights_hidden_to_out.length; wi2hr++) //every row
			{
				for(int wi2hc = 0; wi2hc < weights_hidden_to_out[0].length; wi2hc++) //every column
				{
					writer.append(Double.toString(weights_hidden_to_out[wi2hr][wi2hc]));
					if(!(wi2hc+1 > weights_hidden_to_out[0].length))
					{
						writer.append(",");
					}
				}
				writer.append("\n");
			}
			//ALL BIASES AND WEIGHTS TO FILE.
			writer.flush();
			writer.close();
		}
		catch(IOException e)
		{
		}
		
	}

	
}
