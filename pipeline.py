import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, expr, when, abs as spark_abs, lit, sum as spark_sum
from pyspark.ml.feature import VectorAssembler
from synapse.ml.isolationforest import IsolationForest

def detect_outliers_dq_spark(df, column_name="value", threshold_factor=3):
    """Simple DQ check using mean ± threshold_factor * std in PySpark"""
    stats_df = df.select(
        mean(col(column_name)).alias("mean"),
        stddev(col(column_name)).alias("std")
    ).collect()[0]

    mean_val, std_val = stats_df["mean"], stats_df["std"]
    lower_bound = mean_val - (threshold_factor * std_val)
    upper_bound = mean_val + (threshold_factor * std_val)

    df_out = df.withColumn("dq_outlier",
                           when((col(column_name) < lower_bound) | (col(column_name) > upper_bound), 1).otherwise(0))
    return df_out, (lower_bound, upper_bound)

def detect_outliers_iqr_spark(df, column_name="value", multiplier=1.5):
    """IQR method for outlier detection in PySpark"""
    # Calculate approximate quantiles (0.0 relative error for exactness on small data)
    quantiles = df.stat.approxQuantile(column_name, [0.25, 0.75], 0.0)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1

    lower_bound = q1 - (multiplier * iqr)
    upper_bound = q3 + (multiplier * iqr)

    df_out = df.withColumn("iqr_outlier",
                           when((col(column_name) < lower_bound) | (col(column_name) > upper_bound), 1).otherwise(0))
    return df_out, (lower_bound, upper_bound)

def detect_outliers_zscore_spark(df, column_name="value", threshold=3):
    """Z-score method for outlier detection in PySpark"""
    stats_df = df.select(
        mean(col(column_name)).alias("mean"),
        stddev(col(column_name)).alias("std")
    ).collect()[0]

    mean_val, std_val = stats_df["mean"], stats_df["std"]

    df_out = df.withColumn("z_score", spark_abs((col(column_name) - mean_val) / std_val))
    df_out = df_out.withColumn("z_outlier", when(col("z_score") > threshold, 1).otherwise(0))
    return df_out.drop("z_score"), threshold

def detect_outliers_robust_zscore_spark(df, column_name="value", threshold=3.5):
    """Robust Z-score using median and MAD in PySpark"""
    median = df.stat.approxQuantile(column_name, [0.5], 0.0)[0]

    # Calculate Absolute Deviation from Median
    df_mad = df.withColumn("abs_dev", spark_abs(col(column_name) - median))
    mad = df_mad.stat.approxQuantile("abs_dev", [0.5], 0.0)[0]

    # Avoid division by zero
    mad_safe = mad if mad != 0 else 1e-6

    df_out = df.withColumn("robust_z", lit(0.6745) * (col(column_name) - median) / mad_safe)
    df_out = df_out.withColumn("rz_outlier", when(spark_abs(col("robust_z")) > threshold, 1).otherwise(0))
    return df_out.drop("robust_z"), threshold

def main():
    spark = SparkSession.builder \
        .appName("FinancialOutlierDetectionPipeline") \
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:1.1.2") \
        .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    np.random.seed(101)
    n_normal = 90
    n_fraud = 10

    normal_transactions = {
        'amount': np.random.gamma(shape=2, scale=100, size=n_normal),
        'frequency': np.random.poisson(lam=3, size=n_normal),
        'merchant_cat': np.random.randint(1, 10, size=n_normal),
        'hour': np.random.normal(loc=14, scale=4, size=n_normal).clip(0, 23),
        'geo_distance': np.random.exponential(scale=5, size=n_normal),
        'account_age': np.random.uniform(100, 2000, size=n_normal),
        'is_fraud': [0] * n_normal
    }

    fraud_transactions = {
        'amount': np.array([250, 300, 280, 320, 290, 310, 270, 295, 305, 285]),
        'frequency': np.array([8, 9, 7, 10, 8, 9, 8, 7, 9, 8]),
        'merchant_cat': np.array([9, 9, 9, 9, 9, 9, 9, 9, 9, 9]),
        'hour': np.array([3, 2, 4, 3, 2, 3, 4, 2, 3, 4]),
        'geo_distance': np.array([45, 50, 48, 52, 47, 49, 51, 46, 53, 48]),
        'account_age': np.array([15, 18, 12, 20, 14, 16, 13, 19, 17, 11]),
        'is_fraud': [1] * n_fraud
    }

    pdf_normal = pd.DataFrame(normal_transactions)
    pdf_fraud = pd.DataFrame(fraud_transactions)
    pdf_combined = pd.concat([pdf_normal, pdf_fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_sec5 = spark.createDataFrame(pdf_combined)

    print("--- Dataset Overview ---")
    print(f"Total Transactions: {df_sec5.count()}")
    print(f"Normal: {n_normal}, Fraudulent: {n_fraud}")

    # 1. Univariate Outlier Detection on 'amount'
    df_sec5, _ = detect_outliers_dq_spark(df_sec5, column_name="amount")
    df_sec5, _ = detect_outliers_iqr_spark(df_sec5, column_name="amount")
    df_sec5, _ = detect_outliers_zscore_spark(df_sec5, column_name="amount")
    df_sec5, _ = detect_outliers_robust_zscore_spark(df_sec5, column_name="amount")

    univariate_results = df_sec5.select(
        spark_sum("dq_outlier").alias("DQ"),
        spark_sum("iqr_outlier").alias("IQR"),
        spark_sum("z_outlier").alias("Z-Score"),
        spark_sum("rz_outlier").alias("Robust Z")
    ).collect()[0]

    print("\n--- Univariate Outlier Detection on 'amount' ---")
    print(f"DQ Method: {univariate_results['DQ']} outliers")
    print(f"IQR Method: {univariate_results['IQR']} outliers")
    print(f"Z-Score: {univariate_results['Z-Score']} outliers")
    print(f"Robust Z-Score: {univariate_results['Robust Z']} outliers")

    # 2. Multivariate Isolation Forest
    feature_cols = ['amount', 'frequency', 'merchant_cat', 'hour', 'geo_distance', 'account_age']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_df = assembler.transform(df_sec5)

    iso_forest = IsolationForest(
        featuresCol="features",
        predictionCol="prediction",
        scoreCol="outlierScore",
        contamination=0.1,
        contaminationError=0.01,
        numEstimators=100,
        maxSamples=100,
        maxFeatures=1.0,
        randomSeed=42
    )

    model = iso_forest.fit(assembled_df)
    predictions = model.transform(assembled_df)

    eval_iso = predictions.select(
        spark_sum(when(col("prediction") == 1, 1).otherwise(0)).alias("total_flagged"),
        spark_sum(when((col("prediction") == 1) & (col("is_fraud") == 1), 1).otherwise(0)).alias("true_fraud_caught")
    ).collect()[0]

    total_flagged = eval_iso['total_flagged']
    true_fraud_caught = eval_iso['true_fraud_caught']

    print("\n--- SynapseML Isolation Forest Results ---")
    print(f"Total outliers detected: {total_flagged}")
    print(f"Fraud detected: {true_fraud_caught}/{n_fraud} ({(true_fraud_caught/n_fraud)*100:.0f}%)")
    print(f"False positives: {total_flagged - true_fraud_caught}")

    spark.stop()

if __name__ == "__main__":
    main()
