import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class task_g {

    public static class AccessLogMapper extends Mapper<Object, Text, Text, Text> {

        private Text userId = new Text();
        private Text accessTime = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 5) {
                String byWho = tokens[1];
                String time = tokens[4];
                userId.set(byWho);
                accessTime.set(time);
                context.write(userId, accessTime);
            }
        }
    }

    public static class LinkbookMapper extends Mapper<Object, Text, Text, Text> {

        private Text userId = new Text();
        private Text nickname = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 5) {
                String id = tokens[0];
                String nick = tokens[1];
                userId.set(id);
                nickname.set(nick);
                context.write(userId, nickname);
            }
        }
    }

    public static class reducerdetermine_90 extends Reducer<Text, Text, Text, Text> {
        // Cutoff time - representing 90 days' worth of minutes
       //private int cutoffTime = 1000000 - (90 * 24 * 60);  // Subtract 90 days in minutes from the maximum time
        private int cutoffTime = 870400;

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int latestAccessTime = 0;
            String nickname = "";
            for (Text val : values) {
                String value = val.toString();
                try {
                    int accessTime = Integer.parseInt(value);
                    if (accessTime > latestAccessTime) {
                        latestAccessTime = accessTime;
                    }
                } catch (NumberFormatException e) {
                    nickname = value;
                }
            }
            if (latestAccessTime < cutoffTime && !nickname.isEmpty()) {
                context.write(key, new Text(nickname));
            }
        }

    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: task_g <docker_arg> <accesslog input> <linkbook input> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "expiry linkbook 90 days");

        job.setJarByClass(task_g.class);

        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducerdetermine_90.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public void debug(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: task_g debug <docker_arg> <accesslog input> <linkbook input> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "expiry linkbook 90 days debug");

        job.setJarByClass(task_g.class);

        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducerdetermine_90.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        if (job.waitForCompletion(true)) {
            System.out.println("Debug Job completed successfully.");
        } else {
            System.out.println("Debug Job failed.");
        }
    }
}