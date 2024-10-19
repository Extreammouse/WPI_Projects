import org.junit.Test;

public class task_3_test {

    @Test
    public void debug() throws Exception {
        String[] input = new String[3]; //update this based on output

        /*
        1. put the data.txt into a folder in your pc
        2. add the path for the following two files.
            windows : update the path like "file:///C:/Users/.../projectDirectory/data.txt"
            mac or linux: update the path like "file:///Users/.../projectDirectory/data.txt"
        */

        input[0] = "DOCKER!!";
        input[1] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Dataset/LinkBookPage.csv";
        input[2] = "file:///Users/ehushubhamshaw/Desktop/hadoop-java-code-ds503/Task3/output";

        task_3 t3 = new task_3();
        t3.debug(input);
    }
}