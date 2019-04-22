// Databricks notebook source
// MAGIC %md
// MAGIC #### Q2 - Skeleton Scala Notebook
// MAGIC This template Scala Notebook is provided to provide a basic setup for reading in / writing out the graph file and help you get started with Scala.  Clicking 'Run All' above will execute all commands in the notebook and output a file 'examplegraph.csv'.  See assignment instructions on how to to retrieve this file. You may modify the notebook below the 'Cmd2' block as necessary.
// MAGIC 
// MAGIC #### Precedence of Instruction
// MAGIC The examples provided herein are intended to be more didactic in nature to get you up to speed w/ Scala.  However, should the HW assignment instructions diverge from the content in this notebook, by incident of revision or otherwise, the HW assignment instructions shall always take precedence.  Do not rely solely on the instructions within this notebook as the final authority of the requisite deliverables prior to submitting this assignment.  Usage of this notebook implicitly guarantees that you understand the risks of using this template code. 

// COMMAND ----------

/*
DO NOT MODIFY THIS BLOCK
This assignment can be completely accomplished with the following includes and case class.
Do not modify the %language prefixes, only use Scala code within this notebook.  The auto-grader will check for instances of <%some-other-lang>, e.g., %python
*/
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.functions._
case class edges(Source: String, Target: String, Weight: Int)
import spark.implicits._

// COMMAND ----------

/* 
Create an RDD of graph objects from our toygraph.csv file, convert it to a Dataframe
Replace the 'examplegraph.csv' below with the name of Q2 graph file.
*/

// I changed all of the var's to val as a val is immutable

val df = spark.read.textFile("/FileStore/tables/bitcoinotc.csv") 
  .map(_.split(","))
  .map(columns => edges(columns(0), columns(1), columns(2).toInt)).toDF()

// COMMAND ----------

// Insert blocks as needed to further process your graph, the division and number of code blocks is at your discretion.

// COMMAND ----------

// e.g. eliminate duplicate rows
val data_frame_duplicates_removed = df.dropDuplicates()
data_frame_duplicates_removed.show()


// COMMAND ----------

// e.g. filter nodes by edge weight >= supplied threshold in assignment instructions
val filtered_data_frame = data_frame_duplicates_removed.filter("Weight >= 5")

// COMMAND ----------



// COMMAND ----------

//find node with highest weighted-in-degree, if two or more nodes have the same weighted-in-degree, report the one with the lowest node id
val weighted_in_degree = filtered_data_frame.groupBy("Target")
                                            .sum()
                                            .withColumnRenamed("Target", "Node")
                                            .withColumnRenamed("sum(Weight)", "weighted-in-degree")
                                            .sort(desc("weighted-in-degree"), asc("Node"))
weighted_in_degree.show()

// find node with highest weighted-out-degree, if two or more nodes have the same weighted-out-degree, report the one with the lowest node id
val weighted_out_degree = filtered_data_frame.groupBy("Source")
                                            .sum()
                                            .withColumnRenamed("Source", "Node")
                                            .withColumnRenamed("sum(Weight)", "weighted-out-degree")
                                            .sort(desc("weighted-out-degree"), asc("Node"))
weighted_out_degree.show()

// find node with highest weighted-total degree, if two or more nodes have the same weighted-total-degree, report the one with the lowest node id
val weighted_total_degree = weighted_in_degree.join(weighted_out_degree, "Node")
                                              .withColumn("Total", $"weighted-in-degree"+$"weighted-out-degree")
                                              .sort(desc("Total"), asc("Node"))
weighted_total_degree.show()

// COMMAND ----------

/*
Create a dataframe to store your results
Schema: 3 columns, named: 'v', 'd', 'c' where:
'v' : vertex id
'd' : degree calculation (an integer value.  one row with highest weighted-in-degree, a row w/ highest weighted-out-degree, a row w/ highest weighted-total-degree )
'c' : category of degree, containing one of three string values:
                                                'i' : weighted-in-degree
                                                'o' : weighted-out-degree                                                
                                                't' : weighted-total-degree
- Your output should contain exactly three rows.  
- Your output should contain exactly the column order specified.
- The order of rows does not matter.
                                                
A correct output would be:

v,d,c
4,15,i
2,20,o
2,30,t

whereas:
- Node 2 has highest weighted-out-degree with a value of 20
- Node 4 has highest weighted-in-degree with a value of 15
- Node 2 has highest weighted-total-degree with a value of 30

*/

// COMMAND ----------

// One row with highest weighted-in-degree
val in_degree_max = weighted_in_degree.select("Node","weighted-in-degree")
                                      .limit(1) // Taking the first instance because we already sorted so we will get the highest
                                      .withColumnRenamed("Node", "v") // Getting the Node column and renaming it to "v"
                                      .withColumnRenamed("weighted-in-degree","d") // Getting the weighted-in-degree column and renaming it to "d"
                                      .withColumn("c", lit("i")) // Adding a third column "c" and its value will be "i"
in_degree_max.show()

// COMMAND ----------

// One row with highest weighted-out-degree
val out_degree_max = weighted_out_degree.select("Node", "weighted-out-degree")
                                         .limit(1) // Taking the first instance because we already sorted so we will get the highest
                                         .withColumnRenamed("Node", "v") // Getting the Node column and renaming it to "v"
                                         .withColumnRenamed("weighted-out-degree","d") // Getting the weighted-out-degree column and renaming it to "d"
                                         .withColumn("c", lit("o")) // Adding a third column "c" and its value will be "o"
out_degree_max.show()

// COMMAND ----------

// One row with highest weighted-total-degree
val total_degree_max = weighted_total_degree.select("Node", "Total")
                                            .limit(1) // Taking the first instance because we already sorted so we will get the highest
                                            .withColumnRenamed("Node", "v") // Getting the Node column and renaming it to "v"
                                            .withColumnRenamed("Total","d") // Getting the Total column and renaming it to "d"
                                            .withColumn("c", lit("t")) // Adding a third column "c" and its value will be "t"
total_degree_max.show()

// COMMAND ----------

// Final reults, verify with .show()
val final_results = in_degree_max.union(out_degree_max)
                                 .union(total_degree_max)
final_results.show()



// COMMAND ----------

display(final_results)
