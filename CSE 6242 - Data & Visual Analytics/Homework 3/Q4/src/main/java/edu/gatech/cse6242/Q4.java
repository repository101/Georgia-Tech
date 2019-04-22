package edu.gatech.cse6242;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Q4 {

	public static class DegreeMapper
		extends Mapper<Object, Text, Text, IntWritable>{
			private IntWritable add_one = new IntWritable(1);
			private IntWritable subtract_one = new IntWritable(-1);
			private Text target_node = new Text();
			private Text source_node = new Text();

			public void map(Object key, Text  value, Context context) throws
					IOException, InterruptedException{
						StringTokenizer data = new StringTokenizer(value.toString(), "\r");
						while(data.hasMoreTokens()){
							String[] node = data.nextToken().split("\t");
							source_node.set(node[0]);
							target_node.set(node[1]);
							context.write(source_node, add_one);
							context.write(target_node, subtract_one);
					}
			}
		}

		public static class SecondMapper
			extends Mapper<Object, Text, Text, IntWritable>{
				private IntWritable add_one = new IntWritable(1);
				private Text diff_value = new Text();

				public void map(Object key, Text  value, Context context) throws
						IOException, InterruptedException{
							StringTokenizer data = new StringTokenizer(value.toString(), "\r");
							while(data.hasMoreTokens()){
								String[] node = data.nextToken().split("\t");
								diff_value.set(node[1]);
								context.write(diff_value, add_one);
						}
				}
			}

		public static class IntSumReducer
			extends Reducer<Text, IntWritable, Text, IntWritable>{
				private IntWritable result = new IntWritable();

				public void reduce(Text key, Iterable<IntWritable> values,
					Context context) throws IOException, InterruptedException{
						int sum = 0;
						for (IntWritable val : values){
							sum += val.get();
						}
						result.set(sum);
						context.write(key, result);
					}
			}



	public static void main(String[] args) throws Exception {

		//Path temp_directory = new Path("temp_directory");

		Configuration job_1_configuration = new Configuration();
		Job job_1 = Job.getInstance(job_1_configuration, "Job_1");
		job_1.setJarByClass(Q4.class);
		job_1.setMapperClass(DegreeMapper.class);
		job_1.setCombinerClass(IntSumReducer.class);
		job_1.setReducerClass(IntSumReducer.class);
		job_1.setOutputKeyClass(Text.class);
		job_1.setOutputValueClass(IntWritable.class);

		/* TODO: Needs to be implemented */

		FileInputFormat.addInputPath(job_1, new Path(args[0]));
		FileOutputFormat.setOutputPath(job_1, new Path("temp_directory"));
		boolean success = job_1.waitForCompletion(true);


		if (success){
			Configuration job_2_configuration = new Configuration();

			Job job_2 = Job.getInstance(job_2_configuration, "Job_2");
			job_2.setJarByClass(Q4.class);
			job_2.setMapperClass(SecondMapper.class);
			job_2.setCombinerClass(IntSumReducer.class);
			job_2.setReducerClass(IntSumReducer.class);
			job_2.setOutputKeyClass(Text.class);
			job_2.setOutputValueClass(IntWritable.class);

			FileInputFormat.addInputPath(job_2, new Path("temp_directory"));
			FileOutputFormat.setOutputPath(job_2, new Path(args[1]));



			System.exit(job_2.waitForCompletion(true) ? 0 : 1);

		}

	}
	}
