import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.HashSet;

public class task_2 {

    public static class AccessLogMapper extends Mapper<Object, Text, Text, Text> {
        private Text Bywho = new Text();
        private Text PersonId1 = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 2) {
                String byWho = tokens[1];
                String whatPage = tokens[2];
                Bywho.set(byWho);
                PersonId1.set(whatPage);
                context.write(Bywho, PersonId1);
            }
        }
    }

    public static class LinkbookMapper extends Mapper<Object, Text, Text, Text> {
        private Text ownerId = new Text();
        private Text nickname = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 5) {
                String id = tokens[0];
                String naam = tokens[1];
                ownerId.set(id);
                nickname.set(naam);
                context.write(ownerId, new Text("nickname," + naam));
            }
        }
    }

    public static class reducer extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            boolean accessedOwnPage = false;
            String nickname = "";

            for (Text val : values) {
                String value = val.toString();
                if (value.startsWith("nickname,")) {
                    nickname = value.split(",")[1];
                } else if (value.equals(key.toString())) {
                    accessedOwnPage = true;
                }
            }
            if (accessedOwnPage && !nickname.isEmpty()) {
                context.write(key, new Text(nickname));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Identify Non-Accessed Associates");

        job.setJarByClass(task_2.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[3]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public void debug(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Identify Non-Accessed Associates (Debug)");

        job.setJarByClass(task_2.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        if (job.waitForCompletion(true)) {
            System.out.println("Debug job completed successfully.");
        } else {
            System.out.println("Debug job failed.");
        }
    }
}