import java.util.Random;


public class initializer
{

	public double[] InitializeBias(double[] bias)
	{
		Random rng = new Random();
		rng.nextDouble();
		
		for(int rowindex = 0; rowindex < bias.length; rowindex++)
                {
                        //Will loop 30 times. And there's 30 rows in the weights from input to hidden
                        bias[rowindex] = (-1.0) + (1.0-(-1.0)) * rng.nextDouble(); //Will give a number between -1 and 1.

                }
		
		return bias;
	}



	public double[][] InitializeWeights(double[][] weights, double size1, double size2)
	{
		Random rng = new Random();
		rng.nextDouble();

		for(int rowindex = 0; rowindex < weights.length; rowindex++)
                {
                        for(int columnindex = 0; columnindex < weights[0].length; columnindex++)
                        {
                                weights[rowindex][columnindex] = (-1) + (1-(-1)) * rng.nextDouble(); //ALSO does number between -1 and 1.
                        }
                }
		return weights;
	}



}
