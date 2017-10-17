
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class CSV_import
{

	//FORMAT OF DATA:
	//Columns: 785 columns. First one is expected value in decimal, then 784 inputs for neurons
	//Rows: 60000 rows
	private static int RowSize; //changes
	private static int ColumnSize;
	private static double[][] TheData;

	public static double[][] csvimport(boolean train1test0, boolean isTraining) //PRSize = Row Size, PISize = Index Size
	{

		//Test one. Seatch for mnist_train.csv. Return 1 if found.
		String path;
		//We'll use the version of getProperty that returns a string instead of a propery, done by specifying our want in string format
		//user.dir asks getProperty for the current directory. Concatenate /mnist_train.csv and we've a path to our file (as long as it's)
			//in the same folder.
		if(train1test0 == true)
		{
			path = System.getProperty("user.dir") + "/mnist_train.csv";
			RowSize = 60000;
		}
		else
		{
			path = System.getProperty("user.dir") + "/mnist_test.csv";
			RowSize = 10000;
		}
		//Okay, now we have what I assume to be the path we need to the csv file. We've "located" it.

		//Next we load or read it into...well, whatever.
		//Thanks to mkyong from https://www.mkyong.com/java/how-to-read-and-parse-csv-file-in-java/ for csv parsing code!
		
		BufferedReader br = null; //A BufferedReader object we'll use to perform operations. No constructor...yet.
		String line = ""; //This will hold every line as we go down the whol kit kabootle.
		String cvsSplitBy = ","; //This will be the character by which we split the CSV data. Split by, dare I say it...commas.
		TheData = new double[RowSize][785]; //This will be where all the data from the CSV goes

		//++++++++++++++++ LOCATE AND READ ++++++++++++++++
		try
		{

			br = new BufferedReader(new FileReader(path)); //NOW we run the bufferedreader constructor, opening the file at the path

			int rowindex = 0; //Okay, I'm just gunna jury-rig this so that we can populate the whole thing.

			while ((line = br.readLine()) != null) //So, br.readLine() returns a string. If it returns null, we're at the end of the CSV. This only checks.
			{

				int columnindex = 0; //An embarrassing way to do this but I'm under timecrunch dammit. And make it 0 after every row array pass.

				String[] row = line.split(cvsSplitBy); //We will have here an array of strings for a single line. So it'll be size 785.
				for(String rowdata : row) //This will copy the data into rowdata
				{
					
					TheData[rowindex][columnindex] = Double.parseDouble(rowdata); //Place what's in rowdata into TheData
					//Column index-1 because I only want columns 1 through 748 put in. Thus, indexes 0 through 783 must be used.
					//So to recap, in a dataset with 785 columns, indexed 0 to 748
					//index 0 put into expected. index 1-784 put into data.
					
					columnindex++;

				}
				
				rowindex++; //This will slowly...painfully...place every bit of data in the CSV into a big giant array.
				

			}
			//This has populated TWO 2D arrays.
			br.close(); //Just to clean up.

		}
		catch(FileNotFoundException e) //Error catcher
		{
			 e.printStackTrace(); //Let's see what went down
		}
		catch(IOException e) //Pretty sure this one is just a generic "The system is throwing a hissy fit and I dunno why" error
		{
			e.printStackTrace(); //Yeah, might as well see it.
		}
		finally //Even if we screw the pooch we can still clean up after ourselves
		{
			if (br != null) //So we failed in the middle of this. And the file is open. Oh god.
			{

				try
				{
					br.close(); //tidying up if the system's gone and shit itself
				}
				catch(IOException e) //OH COME ON
				{
					System.out.println("Failed to finish parsing. Failed to close. This is well fucked.");
					e.printStackTrace();
				}

			}
		}

		//++++++++++++++++ SHUFFLE ++++++++++++++++
		//So now we have a TheData[][] filled up with the whole CSV. Now we just need to shuffle it.
		//But, uh...I got a good, accurate gut feeling that the code I have is bunk. Better test and revise until java stops
		//spitting in my face and calling me a son of a whore.
		//That was relatively quick!

		//Okay, some notes:
		//array.length(), when done for a 2D array, returns the number of rows.
		//Our shuffle will be a "bubble shuffle", where items shuffled will be untouched and only unshuffled items will be considered.

		if(isTraining) //No Need to Shuffle if not training.
		{
			TheData = shuffleData(TheData);
		}

		return TheData;
	}

	//The follow shuffle code is adapted from Lars Vogel's code to shuffle a 1D array, URL:
	//http://www.vogella.com/tutorials/JavaAlgorithmsShuffle/article.html
	public static double[][] shuffleData(double[][] Unshuffled)
	{

		int n = Unshuffled.length; //This will return the number of rows. Needed so we don't get that oh so lovely array error you ALWAYS get you know the one
		Random random = new Random(); //Get ourselves a new random object. It's not a random object, it's very decidedly a specific object called random.
		random.nextInt(); //Get random started with its first number.
		
		for (int i = 0; i < n; i++) //Start at 0 and keep it less than n. Because, you know, the whole starting at 0 thing.
		{

			int change = i + random.nextInt(n - i); //Here's the magic. As i steadily increases, the row it will swap with is an offset from i equal
								//to some number in range of 0 to size-i. This creates a bubble shuffle, all rows behind i will
								//be untouched after being shuffled, and i can only swap with rows ahead of itself. The range
								//of possible rows ahead must therefore decrease at the same rate, else you'll be smacked with
								//another array out of bounds error. Again.

			Unshuffled = swap(Unshuffled, i, change);
			

		}
	
		return Unshuffled;

	}

	private static double[][] swap(double[][] Shuffling, int i, int change)
	{

		double[] helper = new double[Shuffling[i].length]; //A quick array, sized as the columns. This will store a row temporarily.
		//populate helper with data from row i
		for(int helpindex = 0; helpindex < helper.length; helpindex++)
		{

			helper[helpindex] = Shuffling[i][helpindex]; //store values of ith row into helper

		}

		//populate Shuffling at row i with data from Shuffling at row change
		for(int helpindex2 = 0; helpindex2 < helper.length; helpindex2++)
		{

			Shuffling[i][helpindex2] = Shuffling[change][helpindex2]; //Copy the data

		}

		//Populate Shuffling at row change with data from helper
		for(int helpindex3 = 0; helpindex3 < helper.length; helpindex3++)
		{

			Shuffling[change][helpindex3] = helper[helpindex3]; //Place in stored data

		}

		return Shuffling; //Send it back. Now all this shit should get destroyed once the method is done.

	}

}
