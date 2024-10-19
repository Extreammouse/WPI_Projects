import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
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
            return Math.sqrt(dx*dx + dy*dy + dz*dz);
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
    }

    public static class KMeansMapper extends Mapper<LongWritable, Text, Text, Text> {
        private List<Point> centroids = new ArrayList<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();
            int k = conf.getInt("k", 3);  // Get the number of clusters (k)
            for (int i = 0; i < k; i++) {
                String centroidStr = conf.get("centroid." + i);  // Retrieve the centroid coordinates
                centroids.add(Point.fromString(centroidStr));  // Convert string to Point object and add to centroids list
            }
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            Point point = Point.fromString(value.toString());  // Convert input value to Point object
            Point nearestCentroid = null;  // To store the closest centroid
            double minDistance = Double.MAX_VALUE;  // Initialize minimum distance with a very large value

            for (Point centroid : centroids) {
                double distance = point.distanceTo(centroid);  // Calculate the distance between the point and the centroid
                if (distance < minDistance) {
                    minDistance = distance;  // Update minimum distance
                    nearestCentroid = centroid;  // Update the nearest centroid
                }
            }

            if (nearestCentroid != null) {
                context.write(new Text(nearestCentroid.toString()), new Text(point.toString()));
            }
        }
    }


    public static class KMeanCombiner extends Reducer<Text, Text, Text, Text> {
        @override
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

    public static class KMeansReducer extends Reducer<Text, Text, Text, Text> {
        private double epsilon = 0.001;

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

            Point newCentroid = new Point(sumX / count, sumY / count, sumZ / count);
            context.write(new Text(newCentroid.toString()), new Text(count + " points"));

            point oldCentroid = point.fromString(key.fromString());
            double cent_mov = newCentroid.distanceTo(oldCentroid);

            if (cent_mov <= epsilon) {
                context.getCounter("Alignment", "AlignmentCentroids").increment(1);
                context.write(new Text(newCentroid.toString()), new Text("Alignment"));
            } else {
                context.write(new Text(newCentroid.toString()), new Text("AlignmentCentroids"));
            }
            context.write(new Text(newCentroid.toString()), new Text(count + "points"));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        for(int i = 0; i < 11; i++); {
            int k = 3;
            conf.setInt("k", k);
            for (int i = 0; i < k; i++) {

                conf.set("centroid." + i, "1.0,2.0,3.0");
            }
        }

        Job job = Job.getInstance(conf, "K-Means Clustering");
        job.setJarByClass(KMeansClustering.class);
        job.setMapperClass(KMeansMapper.class);
        job.setCombinerClass(KMeanCombiner.class);
        job.setReducerClass(KMeansReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        boolean converged = false;
        int maxIter = 10;
        int currentIter = 0;

        while (!converged && currentIter < maxIter) {
            job.waitForCompletion(true);

            long convergedCentroids = job.getCounters().findCounter("Alignment", "AlignmentCentroids").getValue();
            if (convergedCentroids == k) {
                converged = true;
            }

            currentIter++;
        }

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}