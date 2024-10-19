import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;

import java.io.IOException;

public class task_f {

    public static class AssociateMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text id = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 2) {
                String id1 = tokens[0];
                String id2 = tokens[1];
                id.set(id1);
                context.write(id, one);
                id.set(id2);
                context.write(id, one);
            }
        }
    }

    public static class countforrelation extends Reducer<Text, IntWritable, Text, IntWritable> {
        private int totalr = 0;
        private int totalu = 0;
        private int average = totalr / totalu;
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
            totalr += sum;
            totalu++;
        }
        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            context.write(new Text("TOTAL_RELATIONSHIPS"), new IntWritable(totalr));
            context.write(new Text("TOTAL_USERS"), new IntWritable(totalu));
            context.write(new Text("average"), new IntWritable(average));
        }
    }

    public static class averagepopularcalculator extends Reducer<Text, IntWritable, Text, IntWritable> {
        private double averageRelations = 0.0;
        protected void setup(Context context) throws IOException, InterruptedException {
            averageRelations = context.getConfiguration().getDouble("average", 0.0);
        }
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            if (sum > averageRelations) {
                context.write(key, new IntWritable(sum));
            }
        }

    }

    public static void debug(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "average f task debug");
        job.setJarByClass(task_f.class);

        job.setMapperClass(task_f.AssociateMapper.class);
        job.setReducerClass(task_f.countforrelation.class);
        job.setReducerClass(task_f.averagepopularcalculator.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[1])); // Input path
        FileOutputFormat.setOutputPath(job, new Path(args[3])); // Output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
        if (job.waitForCompletion(true)) {
            System.out.println("Job completed successfully.");
        } else {
            System.out.println("Job failed.");
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "average f task");
        job.setJarByClass(task_f.class);

        job.setMapperClass(task_f.AssociateMapper.class);
        job.setReducerClass(task_f.countforrelation.class);
        job.setReducerClass(task_f.averagepopularcalculator.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);

        if (job.waitForCompletion(true)) {
            System.out.println("Job completed successfully.");
        } else {
            System.out.println("Job failed.");
        }
    }
}