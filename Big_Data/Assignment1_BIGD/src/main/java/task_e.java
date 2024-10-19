import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashSet;

public class task_e {

    public static class AssociateMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final static IntWritable one = new IntWritable(1);
        private Text id = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 2) {
                String id1 = tokens[0];
                String id2 = tokens[1];

                // Emit id1 and id2 as keys with a value of 1 to indicate relationships
                id.set(id1);
                context.write(id, one);
                id.set(id2);
                context.write(id, one);
            }
        }
    }

    public static class EducationMapper extends Mapper<Object, Text, Text, Text> {

        private Text bywho = new Text();
        private Text whatpage = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] usrid = value.toString().split(",");
            if (usrid.length >= 3) {
                String ownerbywho = usrid[1];
                String ownerwhatpage = usrid[2];
                bywho.set(ownerbywho);
                whatpage.set(ownerwhatpage);
                context.write(bywho, whatpage);
            }
        }
    }

    public static class EducationReducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            HashSet<String> uniqueaccess = new HashSet<>();

            for (Text page : values) {
                sum++;
                uniqueaccess.add(page.toString());
            }
            String valuee = uniqueaccess + "\t" + uniqueaccess.size();
            context.write(key, new Text(valuee));
        }
    }

    public static void debug(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "accesspage Graduate");
        job.setJarByClass(task_e.class);

        job.setMapperClass(task_e.EducationMapper.class);
        job.setReducerClass(task_e.EducationReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[1])); // Input path
        FileOutputFormat.setOutputPath(job, new Path(args[3])); // Output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "accesspage Users");
        job.setJarByClass(task_e.class);

        job.setMapperClass(task_e.EducationMapper.class);
        job.setReducerClass(task_e.EducationReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}