package com.company;

import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import com.sun.istack.internal.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {

    private static String inputFilePath = "trainBig.arff";
    private static int numOfClusters = 3;
    private static double epsilon = 0.0001;
    private static int numIterations = 0;
    private static int maxIterations = 100;
    private static int seed = 10;
    private static boolean compareToWekaSimpleKMeans = true;
    private static boolean createCharts = false;

    private static Instances clusterCentroids;
    private static Instances initialClusterCentroids;
    private static double[][] ranges;
    private static double[] squaredErrors;
    private static double[] clusterSizes;
    private static double SSE = 0;

    // Data gathered from input file
    private static int numAttributes;
    private static int numInstances;
    private static DataSource data = null;
    private static Instances instances = null;

    private static BufferedWriter writer = null;

    public static void main(String[] args) throws Exception{

        // Handling input arguments
        // java -cp "jcommon.jar:jfreechart.jar:weka.jar:" -jar KMeansClustering.jar 3 0.0001 100 y n trainBig.arff

        for (int i = 0; i < args.length; i++) {
            switch(i){
                case 0:
                    numOfClusters = Integer.parseInt(args[0]);
                    if (numOfClusters <= 0) {
                        System.out.println("Number of clusters cannot be zero. Using default of 3");
                        numOfClusters = 3;
                    }
                    break;
                case 1:
                    epsilon = Double.parseDouble(args[1]);
                    if (epsilon <= 0.0) {
                        System.out.println("Invalid epsilon. Using default of 0.0001");
                        epsilon = 0.0001;
                    }
                    break;
                case 2:
                    maxIterations = Integer.parseInt(args[2]);
                    if (maxIterations <= 0) {
                        System.out.println("Number of max iterations cannot be zero. Using default of 100");
                        maxIterations = 100;
                    }
                    break;
                case 3:
                    compareToWekaSimpleKMeans = args[3].equalsIgnoreCase("y");
                    if (!args[3].equalsIgnoreCase("y") && !args[3].equalsIgnoreCase("n")) {
                        System.out.println("Invalid input for Weka comparison. Using default answer (n).");
                    }
                    break;
                case 4:
                    createCharts = args[4].equalsIgnoreCase("y");
                    if (!args[4].equalsIgnoreCase("y") && !args[3].equalsIgnoreCase("n")) {
                        System.out.println("Invalid input for runtime testing. Using default answer (n).");
                    }
                case 5:
                    inputFilePath = args[5];
                    break;
            }
        }

        grabData(inputFilePath);

        if (data == null || instances == null) {
            System.out.println("Error in reading data. Exiting.");
            return;
        }

        System.out.println("Running program...");

        String outputFilePath = inputFilePath.substring(0,inputFilePath.length()-5) + "Results.txt";
        writer = new BufferedWriter(new FileWriter(outputFilePath, true));
        writer.append("");
        writer.newLine();

        // Run clustering and algorithm
        KMeans();

        if (createCharts) {
            testRuntimeOfProgram();
        }

        writeResults();
        if (writer != null) {
            writer.close();
        }

        System.out.println("Program complete. You'll find the results in " + outputFilePath);
        System.out.println("Recommend NOT opening with regular Notepad. The formatting will be hard to read.");
    }

    /**
     * Given a file path, it'll attempt to grab the necessary data from it.
     * {@link DataSource} allows the reading of files other than ARFF, but it must be appropriately formatted
     *
     * @param fileName - the file path of the data you want to read
     */
    private static void grabData(String fileName) {
        try {
            data = new DataSource(fileName);
            instances = data.getDataSet();
            // setting class attribute if the data format does not provide this information
            // For example, the XRFF format saves the class attribute information as well
            if (instances.classIndex() == -1) {
                instances.setClassIndex(instances.numAttributes() - 1);
            }

            // Number of attributes
            numAttributes = instances.numAttributes();
            // Number of data entries
            numInstances = instances.numInstances();

        } catch (Exception e) {
            System.out.println("Unable to convert data from file. Exiting.");
            System.out.println(e.getLocalizedMessage());
        }
    }

    private static void KMeans() throws Exception {
        Instances newData = new Instances(instances);

        clusterCentroids = new Instances(newData, numOfClusters);
        initialClusterCentroids = new Instances(newData, numOfClusters);
        numIterations = 0;
        squaredErrors = new double[numOfClusters];

        int index;
        double[] oldSSEs;
        int[] clusterAssignments = new int[newData.numInstances()];
        HashMap<Integer, Instance> initialClusters = new HashMap<>();
        Instances[] tempClusters = new Instances[numOfClusters];

        createNormalizationRanges();

        // Randomly choose starting instances/clusters using the default seed of 10
        Random random = new Random(seed);
        for (int i = newData.numInstances() - 1; i >= 0; i--) {
            index = random.nextInt(i+1);
            Instance instance = newData.instance(index);

            if (!initialClusters.containsKey(index)) {
                clusterCentroids.add(instance);
                initialClusters.put(index, instance);
            }

            if (clusterCentroids.numInstances() == numOfClusters) {
                break;
            }
        }

        initialClusterCentroids = clusterCentroids;

        boolean converged = false;
        while (!converged) {
            oldSSEs = squaredErrors.clone();
            squaredErrors = new double[numOfClusters];
            numIterations++;

            for (int i = 0; i < newData.numInstances(); i++) {
                Instance inst = newData.instance(i);
                clusterAssignments[i] = findBestCluster(inst);
            }

            // Creating empty clusters
            clusterCentroids = new Instances(newData, numOfClusters);
            for (int i = 0; i < numOfClusters; i++) {
                tempClusters[i] = new Instances(newData, 0);
            }

            // Adding the instances to their respective clusters
            for (int i = 0; i < newData.numInstances(); i++) {
                tempClusters[clusterAssignments[i]].add(newData.instance(i));
            }

            // Moving the centroids if needed
            for (int i = 0; i < numOfClusters; i++) {
                if (tempClusters[i].numInstances() != 0) {
                    moveCentroid(tempClusters[i]);
                }
            }

            // Checking if the clusters moved enough to consider another iteration or if max iterations has been reached
            converged = (Math.abs(sumOfSquaredErrors(squaredErrors) - sumOfSquaredErrors(oldSSEs)) <= epsilon) || (numIterations >= maxIterations);
        }

        clusterSizes = new double[numOfClusters];
        for (int i = 0; i < numOfClusters; i++) {
            clusterSizes[i] = tempClusters[i].numInstances();
        }

        SSE = sumOfSquaredErrors(squaredErrors);
    }

    private static double[] moveCentroid(Instances insts) {
        double[] values = new double[insts.numAttributes()];

        for (int i = 0; i < insts.numAttributes(); i++) {
            if(insts.attribute(i).isNumeric()) {
                values[i] = mean(insts, i);
            } else if(insts.attribute(i).isNominal()) {
                values[i] = mode(insts, i);
            } else {
                values[i] = 0;
            }
        }

        // Creating instance that would be representative of the cluster's centroid (if the point actually existed there)
        Instance newI = (Instance) insts.firstInstance().copy();
        for (int i = 0; i < values.length; i++) {
            newI.setValue(i, values[i]);
        }
        newI.setDataset(null);
        clusterCentroids.add(newI);

        return values;
    }

    private static int findBestCluster(Instance instance) {
        double minD = Integer.MAX_VALUE;
        int bestCluster = 0;

        for (int i = 0; i < clusterCentroids.numInstances(); i++) {
            double distance = distance(instance, clusterCentroids.instance(i));
            if (distance < minD) {
                minD = distance;
                bestCluster = i;
            }
        }

        squaredErrors[bestCluster] += (Math.pow(minD, 2));

        return bestCluster;
    }

    private static double distance(Instance one, Instance two) {
        double distance = 0;

        for (int i = 0; i < one.numValues() && i < two.numValues(); i++) {
            distance += Math.pow(difference(one,two,i), 2);
        }

        return Math.sqrt(distance);
    }

    private static double difference(Instance one, Instance two, int index) {
        // Grab values at certain indices/attributes of each instance
        double v1 = one.value(index);
        double v2 = two.value(index);

        // Needed for class attribute
        if (instances.attribute(index).type() == Attribute.NOMINAL) {
            return ((int)v1 != (int)v2) ? 1 : 0;
        } else if (instances.attribute(index).type() == Attribute.NUMERIC) {
            return (normalize(v1, index) - normalize(v2, index));
        } else {
            return 0;
        }
    }

    private static double normalize(double norm, int i) {
        return ((norm - ranges[i][0])) / (ranges[i][2]);
    }

    private static void createNormalizationRanges() {
        ranges = new double[numAttributes][3];

        Instance firstInstance = instances.instance(0);
        for (int i = 0; i <  numAttributes; i++) {
            ranges[i][0] = firstInstance.value(i);
            ranges[i][1] = firstInstance.value(i);
            ranges[i][2] = 0.0;
        }

        for (int j = 1; j < numInstances; j++) {
            Instance instance = instances.instance(j);
            for (int i = 0; i <  numAttributes; i++) {
                double v = instance.value(i);
                if (v < ranges[i][0]) {
                    ranges[i][0] = v;
                }
                if (v > ranges[i][1]) {
                    ranges[i][1] = v;
                }
                ranges[i][2] = ranges[i][1] - ranges[i][0];
            }
        }
    }

    private static double sumOfSquaredErrors(double[] sses) {
        double sum = 0;
        for (int i = 0; i < sses.length; i++) {
            sum += sses[i];
        }

        return sum;
    }

    private static double mean(Instances insts, int i) {
        double val = 0;
        for(int j = 0; j < insts.numInstances(); ++j) {
            val += insts.instance(j).value(i);
        }

        return val / insts.numInstances();
    }

    // Needed for the Class attribute because it's nominal
    private static double mode(Instances insts, int i) {
        int[] countOfEachValue = new int[insts.attribute(i).numValues()];

        // Keeps counts of all value counts for the nominal attribute
        for(int j = 0; j < insts.numInstances(); ++j) {
            int v = (int)insts.instance(j).value(i);
            countOfEachValue[v]++;
        }

        // Find max value count for nominal attribute and set val equal to its index/attribute position (it's the MODE)
        int max = 0;
        int val = 0;
        for(int k = 0; k < countOfEachValue.length; k++) {
            if(k == 0 || countOfEachValue[k] > max) {
                max = countOfEachValue[k];
                val = k;
            }
        }

        return (double)val;
    }

    private static void writeResults() throws Exception {
        writer.append("Input file: ").append(inputFilePath);
        writer.newLine();
        writer.append("Number of instances: ").append(String.valueOf(numInstances));
        writer.newLine();
        writer.append("Number of attributes: ").append(String.valueOf(numAttributes));
        writer.newLine();
        writer.append("Number of iterations: ").append(String.valueOf(numIterations));
        writer.newLine();
        writer.append("Number of clusters: ").append(String.valueOf(numOfClusters));
        writer.newLine();
        writer.newLine();
        writer.append("Initial starting points:");
        writer.newLine();
        for (int i = 0; i < initialClusterCentroids.size(); i++) {
            writer.append("Cluster ").append(String.valueOf(i)).append(": ").append(initialClusterCentroids.instance(i).toString());
            writer.newLine();
        }
        writer.newLine();
        writer.append("NOTE: Our implementation and Weka will have the same starting clusters as long as we use the same seed for generating the random indices.");
        writer.newLine();

        writer.newLine();
        writer.append("Final cluster centroids:");
        writer.newLine();
        for (int i = 0; i < initialClusterCentroids.size(); i++) {
            writer.append("Cluster ").append(String.valueOf(i)).append(": ").append(clusterCentroids.instance(i).toString());
            writer.newLine();
        }

        writer.newLine();
        writer.append("Final cluster sizes: ").append(Arrays.toString(clusterSizes));
        writer.newLine();
        writer.append("Individual cluster SSEs: ").append(Arrays.toString(squaredErrors));
        writer.newLine();
        writer.append("Total sum of squared errors: ").append(String.valueOf(SSE));
        writer.newLine();
        writer.newLine();

        if (compareToWekaSimpleKMeans) {
            SimpleKMeans kMeans = new SimpleKMeans();
            kMeans.setDoNotCheckCapabilities(true);
            kMeans.setSeed(seed);
            kMeans.setNumClusters(numOfClusters);
            kMeans.setDistanceFunction(new EuclideanDistance());
            kMeans.buildClusterer(instances);

            writer.append("======================================================");
            writer.newLine();
            writer.append("WEKA");
            writer.newLine();
            writer.append("======================================================");
            writer.newLine();

            writer.append(kMeans.toString());
            writer.append("Difference between our SSE and Weka: ").append(String.valueOf(Math.abs(kMeans.getSquaredError() - SSE)));
            writer.newLine();
            writer.append("========================================================================================================");
            writer.newLine();
            writer.append("END OF PROGRAM RUN");
            writer.newLine();
            writer.append("========================================================================================================");
            writer.newLine();
        }
    }

    private static void testRuntimeOfProgram() throws Exception {
        ConcurrentHashMap<Integer, Double> NumClustersVsRuntime = new ConcurrentHashMap<>();
        ConcurrentHashMap<Integer, Double> NumClustersVsSse = new ConcurrentHashMap<>();

        int maxClusters = 30;
        int minClusters = 1;
        long startTime = 0;
        long endTime = 0;
        double timeInSeconds = 0;

        while (minClusters <= maxClusters) {
            numOfClusters = minClusters;

            // Tracking runtime of KMeans (default epsilon = 0.0001 and maxIterations = 100)
            startTime = System.nanoTime();
            KMeans();
            endTime = System.nanoTime();
            timeInSeconds = ((double) (endTime-startTime)) / 1E9;
            NumClustersVsRuntime.put(numOfClusters, timeInSeconds);
            NumClustersVsSse.put(numOfClusters, SSE);

            minClusters++;
        }

        createRuntimeChart(NumClustersVsRuntime, maxClusters);
        createErrorToClustersChart(NumClustersVsSse, maxClusters);
    }

    private static void createRuntimeChart(ConcurrentHashMap<Integer, Double> NumClustersVsRuntime, int maxClusters) throws IOException {
        DefaultCategoryDataset lineChartData = new DefaultCategoryDataset();

        for (int i = 1; i <= maxClusters; i++) {

            // Plotting runtime vs clusters
            lineChartData.addValue(NumClustersVsRuntime.get(i), "Time", String.format("%d", i));

        }

        JFreeChart lineChartObject = ChartFactory.createLineChart(
                "Runtime vs Num of Clusters","# Clusters",
                "Runtime (seconds)",
                lineChartData, PlotOrientation.VERTICAL,
                false,false,false);

        CategoryPlot plot = lineChartObject.getCategoryPlot();
        NumberAxis range = (NumberAxis)plot.getRangeAxis();
        range.setTickUnit(new NumberTickUnit(0.5));
        range.setLowerBound(0.0);

        lineChartObject.setBorderStroke(new BasicStroke(0.7f));

        int width = 960;    /* Width of the image */
        int height = 720;   /* Height of the image */
        File lineChart = new File( "RuntimeVsClusterNumber.jpeg" );
        ChartUtilities.saveChartAsJPEG(lineChart ,lineChartObject, width ,height);
    }

    private static void createErrorToClustersChart(ConcurrentHashMap<Integer, Double> NumClustersVsSse, int maxClusters) throws IOException {
        DefaultCategoryDataset lineChartData = new DefaultCategoryDataset();

        for (int i = 1; i <= maxClusters; i++) {

            // Plotting runtime vs clusters
            lineChartData.addValue(NumClustersVsSse.get(i), "SSE", String.format("%d", i));

        }

        JFreeChart lineChartObject = ChartFactory.createLineChart(
                "SSE vs Num of Clusters","# Clusters",
                "Sum of Squared Error",
                lineChartData, PlotOrientation.VERTICAL,
                false,false,false);

        CategoryPlot plot = lineChartObject.getCategoryPlot();
        NumberAxis range = (NumberAxis)plot.getRangeAxis();
        range.setLowerBound(0.0);

        lineChartObject.setBorderStroke(new BasicStroke(0.7f));

        int width = 960;    /* Width of the image */
        int height = 720;   /* Height of the image */
        File lineChart = new File( "SseVsClusterNumber.jpeg" );
        ChartUtilities.saveChartAsJPEG(lineChart ,lineChartObject, width ,height);
    }
}
