import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

public class task_3 {

    public static class EducationMapper
            extends Mapper<Object, Text, Text, Text> {

        private Text id = new Text();
        private Text nicknameAndOccupation = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            String[] fields = value.toString().split(",");
            if (fields.length >= 5) {
                String userId = fields[0];
                String userNickname = fields[1];
                String userOccupation = fields[2];
                String highestEdu = fields[4];

                if (highestEdu.equalsIgnoreCase("Masters")) {
                    id.set(userId);
                    nicknameAndOccupation.set(userNickname + "," + userOccupation);
                    context.write(id, nicknameAndOccupation);
                }
            }
        }
    }

    public static class EducationReducer
            extends Reducer<Text, Text, Text, Text> {

        public void reduce(Text key, Iterable<Text> values,
                           Context context) throws IOException, InterruptedException {
            for (Text val : values) {
                context.write(key, val);
            }
        }
    }

    public static void debug(String[] args) throws Exception {

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "LinkBookPage Graduate");
        job.setJarByClass(task_3.class);

        job.setMapperClass(EducationMapper.class);
        job.setReducerClass(EducationReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[1])); // Input path
        FileOutputFormat.setOutputPath(job, new Path(args[2])); // Output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "LinkBookPage grade Users");
        job.setJarByClass(task_3.class);

        job.setMapperClass(EducationMapper.class);
        job.setReducerClass(EducationReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0])); // Input path
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // Output path

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}