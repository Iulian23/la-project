from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, day, month, count, lit, countDistinct
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from pyspark.sql import functions as F
import pandas as pd

# 1. Initialize Spark session
spark = SparkSession.builder \
    .appName("Daily Pressure Analysis") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# 2. Load the data
file_path = '/content/daily_pressure_summary.csv'
df = spark.read.option("header", "true").csv(file_path, inferSchema=True)

# Identify columns with only one unique value
single_value_cols = [col_name for col_name in df.columns if df.select(col_name).distinct().count() == 1]

print("Columns with only one unique value:")
print(single_value_cols)

# Drop the columns
df = df.drop(*single_value_cols)

# Display remaining columns
print("\nRemaining columns after dropping single-value columns:")
df.printSchema()

df = df.drop("local_site_name").dropna()

# Display any NULL columns
cols = [F.col(c) for c in df.columns]
filter_expr = reduce(lambda a, b: a | b.isNull(), cols[1:], cols[0].isNull())

df.filter(filter_expr).show()

df.show()

# Convert date to datetime
df = df.withColumn("date_local", to_date(col("date_local"), "yyyy-MM-dd"))
df = df.withColumn("day_local", day(col("date_local"))).withColumn("month_local", month(col("date_local")))

# Convert date to datetime
df = df.withColumn("date_of_last_change", to_date(col("date_of_last_change"), "yyyy-MM-dd"))
df = df.withColumn("day_last_change", day(col("date_of_last_change"))).withColumn("month_last_change", month(col("date_of_last_change")))

# Convert Spark DataFrame to Pandas DataFrame for plotting
pandas_df = df.toPandas()


#1: Histogram of a numerical column
plt.figure(figsize=(10, 6))
sns.histplot(pandas_df['arithmetic_mean'], kde=True)
plt.title('Distribution of Average Pressure')
plt.xlabel('Average Pressure')
plt.ylabel('Frequency')
plt.show()

#2: Scatter plot of two numerical columns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='day_local', y='arithmetic_mean', data=pandas_df)
plt.title('Average Pressure vs. Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Average Pressure')
plt.show()

#3: Boxplot of a numerical column grouped by a categorical column
plt.figure(figsize=(10, 6))
sns.boxplot(x='month_local', y='arithmetic_mean', data=pandas_df)
plt.title('Average Pressure by Month')
plt.xlabel('Month')
plt.ylabel('Average Pressure')
plt.show()

# Initialize Spark session
spark = SparkSession.builder.appName("AirPressurePrediction").getOrCreate()

# Step 1: Feature Engineering
# Select features and target variable
feature_columns = ["day_local", "month_local", "first_max_value", "first_max_hour"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df_features = assembler.transform(df)

# Step 2: Splitting the data
# Split into training (80%) and testing (20%) datasets
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

# Step 3: Train the regression model
lr = LinearRegression(featuresCol="features", labelCol="arithmetic_mean")
lr_model = lr.fit(train_data)

# Print model coefficients and intercept
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Step 4: Make predictions on the test dataset
predictions = lr_model.transform(test_data)

# Step 5: Evaluate the model
evaluator = RegressionEvaluator(
    labelCol="arithmetic_mean",
    predictionCol="prediction",
    metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 6: Evaluate the model
evaluator = RegressionEvaluator(
    labelCol="arithmetic_mean",
    predictionCol="prediction",
    metricName="r2"
)
r2 = evaluator.evaluate(predictions)
print(f"R2: {r2}")

# Step 7: Display predictions
predictions.select("day_local", "month_local", "arithmetic_mean", "prediction").show()

# Step 1: Convert Predictions to Pandas
predictions_pd = predictions.select("arithmetic_mean", "prediction").toPandas()

# Step 2: Plot Predictions vs Actual Values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_pd["arithmetic_mean"], predictions_pd["prediction"], alpha=0.6, color="blue", label="Predictions")
plt.plot([predictions_pd["arithmetic_mean"].min(), predictions_pd["arithmetic_mean"].max()],
         [predictions_pd["arithmetic_mean"].min(), predictions_pd["arithmetic_mean"].max()],
         color="red", linestyle="--", label="Perfect Fit Line")

# Step 3: Add Labels and Legend
plt.xlabel("Actual Values (arithmetic_mean)")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Predictions vs Actual")
plt.legend()
plt.grid(alpha=0.3)

# Step 4: Show the Plot
plt.show()
