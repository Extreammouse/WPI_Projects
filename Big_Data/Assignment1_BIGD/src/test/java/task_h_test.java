import org.junit.Test;

public class task_h_test {

    @Test
    public void debug() throws Exception {
        String[] input = new String[5]; //update this based on output

        /*
        1. put the data.txt into a folder in your pc
        2. add the path for the following two files.
            windows : update the path like "file:///C:/Users/.../projectDirectory/data.txt"
            mac or linux: update the path like "file:///Users/.../projectDirectory/data.txt"
        */
        input[0] = "DOCKER!!";
        input[1] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/LinkBookPage.csv";
        input[2] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/Associates.csv";
        input[3] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/AccessLogs.csv";
        input[4] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Taskh/output";
//        input[0] = "DOCKER!!";
//        input[1] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/LinkBookPage_small.csv";
//        input[2] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/Associates_small.csv";
//        input[3] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/AccessLogs_small.csv";
//        input[4] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Taskh/output";

        task_h th = new task_h();
        th.debug(input);
    }
}