import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class winePre {
	
    private static final String[] COL_NAMES = new String[] {
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality"
        };
    
    private static Dataset < Row > getTestData(SparkSession spark, String testPath) {
        Dataset < Row > valid_ds = spark.read()
            .option("header", true)
            .option("inferSchema", "true")
            .option("delimiter", ";")
            .csv(testPath);

        int i = 0;
        for (String column: valid_ds.columns()) {
            valid_ds = valid_ds.withColumnRenamed(column, COL_NAMES[i]);
            i++;
        }

        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[] {
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        }).setOutputCol("features");

        Dataset < Row > testDataset = vectorAssembler.transform(valid_ds)
            .select("quality", "features")
            .withColumnRenamed("quality", "label");

        return testDataset;
    }
    
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("quality-prediction")
                .master("local")
				.config("spark.executor.memory", "2147480000").config("spark.driver.memory", "2147480000")
				.config("spark.testing.memory", "2147480000")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        
        String test_path = args[0];
        String model_path = args[1];

        PipelineModel pipelineModel = PipelineModel.load(model_path);
        Dataset<Row> testDf = getTestData(spark, test_path).cache();
        Dataset<Row> results = pipelineModel.transform(testDf).cache();
        results.select("features", "label", "prediction").show(5, false);
        System.out.println();
        
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

        evaluator.setMetricName("accuracy");
        double accuracy1 = evaluator.evaluate(results);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(results);

        System.out.println("Accuracy: " + accuracy1);
        System.out.println("F1: " + f1);
    }
}