import java.io.IOException;
import java.util.HashSet;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Reducer;

public class task_h {

    public static class AssociateMapper extends Mapper<Object, Text, Text, Text> {
        private Text personId1 = new Text();
        private Text personId2 = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 2) {
                String id1 = tokens[0];
                String id2 = tokens[1];
                personId1.set(id1);
                personId2.set(id2);
                context.write(personId1, new Text("associate," + id2));
            }
        }
    }

    public static class AccessLogMapper extends Mapper<Object, Text, Text, Text> {
        private Text byWho = new Text();
        private Text whatPage = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] tokens = value.toString().split(",");
            if (tokens.length >= 3) {
                String byWhom = tokens[1];
                String whatPageId = tokens[2];
                byWho.set(byWhom);
                whatPage.set(whatPageId);
                context.write(byWho, new Text("access," + whatPageId));
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
                String nick = tokens[1];
                ownerId.set(id);
                nickname.set(nick);
                context.write(ownerId, new Text("nickname," + nick));  // Emit: owner_id -> nickname
            }
        }
    }

    public static class reducerforsnake extends Reducer<Text, Text, Text, Text> {
        private HashSet<String> accessedPages = new HashSet<>();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            HashSet<String> associates = new HashSet<>();
            String nickname = "";
            accessedPages.clear();

            for (Text val : values) {
                String[] valueParts = val.toString().split(",");
                String type = valueParts[0];
                String value = valueParts[1];

                if (type.equals("associate")) {
                    associates.add(value);
                } else if (type.equals("access")) {
                    accessedPages.add(value);
                } else if (type.equals("nickname")) {
                    nickname = value;
                }
            }

            for (String associate : associates) {
                if (!accessedPages.contains(associate) && !nickname.isEmpty()) {
                    context.write(key, new Text(nickname));
                }
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: task_h <docker_arg> <linkbook_input> <associates_input> <accesslogs_input> <output_path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Identify Non-Accessed Associates");

        job.setJarByClass(task_h.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AssociateMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[3]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducerforsnake.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[4]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    public void debug(String[] args) throws Exception {
        if (args.length != 5) {
            System.err.println("Usage: task_h debug <docker_arg> <linkbook_input> <associates_input> <accesslogs_input> <output_path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Identify Non-Accessed Associates (Debug)");

        job.setJarByClass(task_h.class);

        MultipleInputs.addInputPath(job, new Path(args[1]), TextInputFormat.class, LinkbookMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[2]), TextInputFormat.class, AssociateMapper.class);
        MultipleInputs.addInputPath(job, new Path(args[3]), TextInputFormat.class, AccessLogMapper.class);

        job.setReducerClass(reducerforsnake.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileOutputFormat.setOutputPath(job, new Path(args[4]));

        if (job.waitForCompletion(true)) {
            System.out.println("Debug job completed successfully.");
        } else {
            System.out.println("Debug job failed.");
        }
    }
}