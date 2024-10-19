import java.io.IOException;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Reducer;

public class task_4 {

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

    public static class LinkbookMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable zero = new IntWritable(0);
        private Text id = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 1) {
                String ownerId = tokens[0];
                id.set(ownerId);
                context.write(id, zero);
            }
        }
    }

    public static class HappinessReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            boolean hasRelationships = false;

            for (IntWritable val : values) {
                if (val.get() > 0) {
                    hasRelationships = true;
                }
                sum += val.get();
            }

            context.write(key, new IntWritable(hasRelationships ? sum : 0));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "happiness factor");

        job.setJarByClass(task_4.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AssociateMapper.class);
        job.setReducerClass(HappinessReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public void debug(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "happiness factor debug");

        job.setJarByClass(task_4.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AssociateMapper.class);

        job.setReducerClass(HappinessReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        if (job.waitForCompletion(true)) {
            System.out.println("Job completed successfully.");
        } else {
            System.out.println("Job failed.");
        }
    }
}