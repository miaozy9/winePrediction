import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class Wine {
    private static final String TRAINING_DATASET = "TrainingDataset.csv";
    private static final String VALIDATION_DATASET = "ValidationDataset.csv";

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

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
            .appName("Wine Quality Prediction")
            .master("local")
            .getOrCreate();

        spark.sparkContext().setLogLevel("ERROR");
        new Wine().logRegTrain(spark);
    }

    private Dataset < Row > getTrainData(SparkSession spark) {
        Dataset < Row > trainData = spark.read()
            .option("header", true)
            .option("inferSchema", "true")
            .option("delimiter", ";")
            .csv(TRAINING_DATASET);

        int i = 0;
        for (String column: trainData.columns()) {
            trainData = trainData.withColumnRenamed(column, COL_NAMES[i]);
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

        Dataset < Row > inputDataset = vectorAssembler.transform(trainData)
            .select("quality", "features")
            .withColumnRenamed("quality", "label");

        return inputDataset;
    }

    private Dataset < Row > getTestData(SparkSession spark) {
        Dataset < Row > valData = spark.read()
            .option("header", true)
            .option("inferSchema", "true")
            .option("delimiter", ";")
            .csv(VALIDATION_DATASET);

        int i = 0;
        for (String column: valData.columns()) {
            valData = valData.withColumnRenamed(column, COL_NAMES[i]);
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

        Dataset < Row > testDataset = vectorAssembler.transform(valData)
            .select("quality", "features")
            .withColumnRenamed("quality", "label");

        return testDataset;
    }

    public void randomForestTrain(SparkSession spark) {
        Dataset < Row > inputDataset = getTrainData(spark);
        Dataset < Row > testDataset = getTestData(spark);

        RandomForestClassifier model = new RandomForestClassifier()
            .setLabelCol("label")
            .setFeaturesCol("features");

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {
            model
        });
        PipelineModel p_model = pipeline.fit(inputDataset);

        Dataset < Row > predictions = p_model.transform(testDataset);

        // Testing and Evaluation 
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Random Forest Accuracy: " + accuracy);

        try {
        	p_model.write().overwrite().save("./model");
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    public void logRegTrain(SparkSession spark) {
        Dataset < Row > inputDataset = getTrainData(spark);
        LogisticRegression logReg = new LogisticRegression();

        Pipeline pl1 = new Pipeline();
        pl1.setStages(new PipelineStage[] {
            logReg
        });

        PipelineModel model1 = pl1.fit(inputDataset);

        Dataset < Row > testDataset = getTestData(spark);
        Dataset < Row > results = model1.transform(testDataset);
        System.out.println("\n Validation Training Set Metrics");
        results.select("features", "label", "prediction").show(5, false);
        
        System.out.println();
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();

        evaluator.setMetricName("accuracy");
        double accuracy1 = evaluator.evaluate(results);

        evaluator.setMetricName("f1");
        double f1 = evaluator.evaluate(results);

        System.out.println("Accuracy: " + accuracy1);
        System.out.println("F1: " + f1);
        try {
            model1.write().overwrite().save("./model");
        } catch (Exception e) {
            System.out.println(e);
        }
    }
}