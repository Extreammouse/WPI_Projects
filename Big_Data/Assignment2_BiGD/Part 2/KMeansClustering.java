package project2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeansClustering {

    public static class Point {
        double x, y, z;

        public Point(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public double distanceTo(Point other) {
            double dx = x - other.x;
            double dy = y - other.y;
            double dz = z - other.z;
            return Math.sqrt(dx * dx + dy * dy + dz * dz);
        }

        @Override
        public String toString() {
            return x + "," + y + "," + z;
        }

        public static Point fromString(String s) {
            String[] parts = s.split(",");
            return new Point(Double.parseDouble(parts[0]),
                    Double.parseDouble(parts[1]),
                    Double.parseDouble(parts[2]));
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (!(obj instanceof Point)) return false;
            Point other = (Point) obj;
            return Double.compare(this.x, other.x) == 0 &&
                   Double.compare(this.y, other.y) == 0 &&
                   Double.compare(this.z, other.z) == 0;
        }
    }

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {
        private List<Point> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            int k = conf.getInt("k", 5);
            centroids.clear();
            for (int i = 0; i < k; i++) {
                String centroidStr = conf.get("centroid." + i);
                if (centroidStr != null) {
                    centroids.add(Point.fromString(centroidStr));
                }
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Point point = Point.fromString(value.toString());
            Point nearestCentroid = null;
            double minDistance = Double.MAX_VALUE;

            for (Point centroid : centroids) {
                double distance = point.distanceTo(centroid);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestCentroid = centroid;
                }
            }

            if (nearestCentroid != null) {
                context.write(new Text(nearestCentroid.toString()), new Text(point.toString()));
            }
        }
    }

    public static class KMeansCombiner extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int count = 0;
            double sumX = 0, sumY = 0, sumZ = 0;

            for (Text value : values) {
                Point point = Point.fromString(value.toString());
                sumX += point.x;
                sumY += point.y;
                sumZ += point.z;
                count++;
            }
            context.write(key, new Text(sumX + "," + sumY + "," + sumZ + "," + count));
        }
    }

    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0, sumY = 0, sumZ = 0;
            int count = 0;

            for (Text value : values) {
                String[] parts = value.toString().split(",");
                if (parts.length == 4) {
                    sumX += Double.parseDouble(parts[0]);
                    sumY += Double.parseDouble(parts[1]);
                    sumZ += Double.parseDouble(parts[2]);
                    count += Integer.parseInt(parts[3]);
                }
            }

            if (count > 0) {
                double newX = sumX / count;
                double newY = sumY / count;
                double newZ = sumZ / count;
                context.write(new Text(newX + "," + newY + "," + newZ), new Text(count + " points"));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        if (args.length < 2) {
            System.err.println("Usage: KMeansClustering <input path> <output path>");
            System.exit(1);
        }

        String inputPath = args[0];
        String outputPath = args[1];

        System.out.println("Input path: " + inputPath);
        System.out.println("Output path: " + outputPath);

        int k = 5;
        conf.setInt("k", k);
        initializeCentroids(conf, new Path(inputPath), k);

        int maxIterations = conf.getInt("max.iterations", 10);
        boolean earlyTermination = conf.getBoolean("early.termination", true);
        double epsilon = conf.getDouble("convergence.threshold", 0.001); // Define epsilon

        boolean converged = false;
        int currentIteration = 0;

        while (currentIteration < maxIterations && (!earlyTermination || !converged)) {
            Job job = Job.getInstance(conf, "K-Means Clustering - Iteration " + currentIteration);
            job.setJarByClass(KMeansClustering.class);
            job.setMapperClass(KMeansMapper.class);
            job.setReducerClass(KMeansReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            job.setCombinerClass(KMeansCombiner.class);

            FileInputFormat.addInputPath(job, new Path(inputPath));
            Path iterationOutputPath = new Path(outputPath + "/iteration_" + currentIteration);

            if (fs.exists(iterationOutputPath)) {
                fs.delete(iterationOutputPath, true);
            }

            FileOutputFormat.setOutputPath(job, iterationOutputPath);

            boolean success = job.waitForCompletion(true);
            if (!success) {
                System.err.println("Job failed for iteration " + currentIteration);
                break;
            }

            // Check for convergence
            boolean allCentroidsConverged = updateCentroidsAndCheckConvergence(conf, iterationOutputPath, k, epsilon);

            // Task a: Single-Iteration K-Means (R=1)
            if (currentIteration == 0) {
                System.out.println("=== Task (a): Single-Iteration K-Means (R=1) ===");
                printCentroids(conf, k);
            }

            // Task b: Basic Multi-Iteration K-Means (R=10)
            if (currentIteration == 9) {
                System.out.println("=== Task (b): Basic Multi-Iteration K-Means (R=10) ===");
                printCentroids(conf, k);
            }

            // Task c: Advanced Multi-Iteration K-Means with Early Termination
            if (allCentroidsConverged) {
                System.out.println("=== Task (c): Multi-Iteration K-Means with Early Termination ===");
                System.out.println("Converged after " + (currentIteration + 1) + " iterations.");
                printCentroids(conf, k);
                converged = true;
                break;
            } else {
                System.out.println("=== Task (c): Multi-Iteration K-Means with Early Termination ===");
                System.out.println("No Convergence.");
                printCentroids(conf, k);

            }

            currentIteration++;
        }

        // Task d: Advanced K-Means with Hadoop MapReduce Combiner
        System.out.println("=== Task (d): K-Means with Hadoop MapReduce Combiner ===");
        printCentroids(conf, k);

        // Task e: Final Output of Clustered Points
        System.out.println("=== Task (e): Final Output of Clustered Points ===");
        printClusteredPoints(conf, k, new Path(outputPath + "/iteration_" + (currentIteration - 1)));
    }

    private static void initializeCentroids(Configuration conf, Path inputPath, int k) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(inputPath)));
        List<Point> randomPoints = new ArrayList<>();

        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            randomPoints.add(new Point(Double.parseDouble(parts[0]), Double.parseDouble(parts[1]), Double.parseDouble(parts[2])));
            if (randomPoints.size() >= k * 10) {
                break;
            }
        }
        br.close();

        for (int i = 0; i < k; i++) {
            int randomIndex = (int) (Math.random() * randomPoints.size());
            conf.set("centroid." + i, randomPoints.get(randomIndex).toString());
        }
    }

    private static boolean updateCentroidsAndCheckConvergence(Configuration conf, Path outputPath, int k, double epsilon) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path outputFile = new Path(outputPath, "part-r-00000");

        if (!fs.exists(outputFile)) {
            System.err.println("Output file does not exist: " + outputFile);
            return false;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(outputFile)));

        String line;
        int i = 0;
        boolean allCentroidsConverged = true;
        while ((line = br.readLine()) != null && i < k) {
            String[] parts = line.split("\t");
            if (parts.length > 0) {
                String newCentroidStr = parts[0];
                String oldCentroidStr = conf.get("centroid." + i);
                
                if (oldCentroidStr != null) {
                    Point newCentroid = Point.fromString(newCentroidStr);
                    Point oldCentroid = Point.fromString(oldCentroidStr);
                    
                    double distance = newCentroid.distanceTo(oldCentroid);
                    if (distance > epsilon) {
                        allCentroidsConverged = false;
                    }
                }
                
                conf.set("centroid." + i, newCentroidStr);
                System.out.println("Centroid " + i + ": " + newCentroidStr);
                i++;
            }
        }
        br.close();
        
        return allCentroidsConverged;
    }

    private static void printCentroids(Configuration conf, int k) {
        System.out.println("=== Centroids ===");
        for (int i = 0; i < k; i++) {
            String centroid = conf.get("centroid." + i);
            System.out.println("Centroid " + i + ": " + centroid);
        }
    }

    private static void printClusteredPoints(Configuration conf, int k, Path outputPath) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path outputFile = new Path(outputPath, "part-r-00000");

        if (!fs.exists(outputFile)) {
            System.err.println("Output file does not exist: " + outputFile);
            return;
        }

        BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(outputFile)));
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\t");
            System.out.println("Cluster Center: " + parts[0] + " | Number of Points: " + parts[1]);
        }
        br.close();
    }
}