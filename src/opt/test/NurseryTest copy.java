package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import opt.prob.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying nursery applicants as either
 * being accepted or not. 
 *
 * @author Devan Stormont
 * @version 1.0
 */
/*
public class NurseryTest {
    private static Instance[] instances = initializeInstances();

	private static final int numSamples = 3888;
    private static int inputLayer = 27, hiddenLayer = 5, outputLayer = 1, trainingIterations = 3000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3]; //[4];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3]; //[4];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3]; //[4];
    private static String[] oaNames = {"RHC", "SA", "GA" }; //, "MIMIC"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);
        //oa[3] = new MIMIC(numSamples, 30, nnop[3]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            int iterations = train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": " +
            			"\nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances." +
                        "\nPercent correctly classified: " + df.format(correct/(correct+incorrect)*100) + "%" +
                        "\nTraining time: " + df.format(trainingTime) + " seconds" +
                        "\nTesting time: " + df.format(testingTime) + " seconds" +
                        "\nIterations: " + iterations +
                        "\n";
        }

        System.out.println(results);
    }

    private static int train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        double start = System.nanoTime();
        int i = 0;

        for(; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

			double current = System.nanoTime();
			double elapsed = (current - start) / Math.pow(10,9);
            System.out.println(i + "," + df.format(error) + "," + elapsed);
            
            if (error < 0.5) {
            	break;
            }
        }
        
        return i;
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[numSamples][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/nursery_transformed.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[inputLayer];
                attributes[i][1] = new double[1];

                for(int j = 0; j < inputLayer; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}
*/