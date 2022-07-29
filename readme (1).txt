Parallel Training part:
Create EMR cluster(4 nodes) spark version
Connect to master node
Download data, training program files from s3
Upload necessary datasets to slave nodes
Run the training program on master node
Retrieve the model generated from hdfs to local
Upload the model to s3 bucket recursively




ssh -i "wine.pem" hadoop@ec2-35-172-116-172.compute-1.amazonaws.com
aws s3 cp s3://miaowineproject/TrainingDataset.csv .
aws s3 cp s3://miaowineproject/ValidationDataset.csv .
aws s3 cp s3://miaowineproject/wineTrain.jar .
hadoop fs -put /home/hadoop/TrainingDataset.csv /user/hadoop/TrainingDataset.csv
hadoop fs -put /home/hadoop/ValidationDataset.csv /user/hadoop/ValidationDataset.csv
spark-submit wineTrain.jar
hdfs dfs -get model
aws s3 cp model s3://miaowineproject/model --recursive                //upload model to s3






Predict part:
Set Access key and secret access key for s3 usage
Download the model from the s3
Download or upload the test data
Run the prediction program


aws configure
mkdir model
cd model
aws s3 cp s3://miaowineproject/model .  --recursive
cd
aws s3 cp s3://miaowineproject/test.csv .                //or upload with ssh manually
java -jar pre.jar test.csv model