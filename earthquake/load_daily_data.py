import os
import json
import requests
from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, ArrayType, FloatType, LongType
from pyspark.sql.functions import from_unixtime, col, current_timestamp

spark = SparkSession.builder.master("local[*]").appName("Earthquake").getOrCreate()

if __name__ == '__main__':

    # Service Key
    os.environ[
        'GOOGLE_APPLICATION_CREDENTIALS'] = (r"D:\GCP-2024\Project\project_earthquake_data\earthquake\earthquake"
                                             r"-project-442611-de6e3a84a75f.json")

    # Step - 1 : Download the data from the URL

    url = r"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

    response = requests.get(url)
    data = response.json()

    # Step - 2 : Set up Google Cloud Storage client

    client = storage.Client()

    # Set the Bucket name

    bucket_name = 'earth_earthquake_data'
    bucket = client.bucket(bucket_name)

    # Step - 3 : Upload the data to the GCS Bucket
    # Convert data to JSON String and then to bytes

    json_data = json.dumps(data)
    blob = bucket.blob('Daily_data_pyspark/bronze_layer/raw_data')  # Specify the path in the bucket
    blob.upload_from_string(json_data, content_type='application/json')

    # ---------------------------------------------------------------------------------------------------------------------------------

    for feature in data['features']:
        # Convert properties to appropriate types
        feature['properties']['mag'] = float(feature['properties']['mag']) if feature['properties'][
                                                                                  'mag'] is not None else None
        feature['properties']['gap'] = float(feature['properties']['gap']) if feature['properties'][
                                                                                  'gap'] is not None else None
        feature['properties']['dmin'] = float(feature['properties']['dmin']) if feature['properties'][
                                                                                    'dmin'] is not None else None
        feature['properties']['rms'] = float(feature['properties']['rms']) if feature['properties'][
                                                                                  'rms'] is not None else None
        feature['properties']['sig'] = int(feature['properties']['sig']) if feature['properties'][
                                                                                'sig'] is not None else None
        feature['properties']['tsunami'] = int(feature['properties']['tsunami']) if feature['properties'][
                                                                                        'tsunami'] is not None else None
        feature['properties']['nst'] = int(feature['properties']['nst']) if feature['properties'][
                                                                                'nst'] is not None else None
        feature['properties']['time'] = int(feature['properties']['time']) if feature['properties'][
                                                                                  'time'] is not None else None
        feature['properties']['updated'] = int(feature['properties']['updated']) if feature['properties'][
                                                                                        'updated'] is not None else None
        feature['properties']['felt'] = int(feature['properties']['felt']) if feature['properties'][
                                                                                  'felt'] is not None else None

        # Convert geometry coordinates to floats

        feature['geometry']['coordinates'] = [float(coord) for coord in feature['geometry']['coordinates']] if \
            feature['geometry']['coordinates'] is not None else []

    # Define the schema

    schemas = StructType([
        StructField("type", StringType(), True),
        StructField("properties", StructType([
            StructField("mag", FloatType(), True),
            StructField("place", StringType(), True),
            StructField("time", LongType(), True),
            StructField("updated", LongType(), True),
            StructField("tz", StringType(), True),
            StructField("url", StringType(), True),
            StructField("detail", StringType(), True),
            StructField("felt", StringType(), True),
            StructField("cdi", StringType(), True),
            StructField("mmi", StringType(), True),
            StructField("alert", StringType(), True),
            StructField("status", StringType(), True),
            StructField("tsunami", IntegerType(), True),
            StructField("sig", IntegerType(), True),
            StructField("net", StringType(), True),
            StructField("code", StringType(), True),
            StructField("ids", StringType(), True),
            StructField("sources", StringType(), True),
            StructField("types", StringType(), True),
            StructField("nst", IntegerType(), True),
            StructField("dmin", FloatType(), True),
            StructField("rms", FloatType(), True),
            StructField("gap", FloatType(), True),
            StructField("magType", StringType(), True),
            StructField("type", StringType(), True),
            StructField("title", StringType(), True)
        ]), True),
        StructField("geometry", StructType([
            StructField("type", StringType(), True),
            StructField("coordinates", ArrayType(FloatType()), True)
        ]), True)
    ])

    # Create DataFrame from fetched data

    df = spark.createDataFrame(data['features'], schema=schemas)
    # df.show(truncate=False)
    # df.printSchema()

    # Flatten the DataFrame

    flattened_df = df.select(
        "properties.mag",
        "properties.place",
        "properties.time",
        "properties.updated",
        "properties.tz",
        "properties.url",
        "properties.detail",
        "properties.felt",
        "properties.cdi",
        "properties.mmi",
        "properties.alert",
        "properties.status",
        "properties.tsunami",
        "properties.sig",
        "properties.net",
        "properties.code",
        "properties.ids",
        "properties.sources",
        "properties.types",
        "properties.nst",
        "properties.dmin",
        "properties.rms",
        "properties.gap",
        "properties.magType",
        "properties.title",
        "geometry.coordinates"
    )

    # Extract longitude, latitude, and depth from the 'coordinates' field

    flattened_df = flattened_df.withColumn("longitude", flattened_df["coordinates"].getItem(0)) \
        .withColumn("latitude", flattened_df["coordinates"].getItem(1)) \
        .withColumn("depth", flattened_df["coordinates"].getItem(2))

    # Drop the 'coordinates' column
    flattened_df = flattened_df.drop("coordinates")

    # Show the final flattened DataFrame

    # flattened_df.show(truncate=False)

    # define datetime for data
    from datetime import datetime

    current_date = datetime.now().strftime("%Y%m%d")

    flattened_df = flattened_df.withColumn("time", from_unixtime(col("time") / 1000).cast("timestamp")).withColumn(
        "updated", from_unixtime(col("updated") / 1000).cast("timestamp"))
    # flattened_df.show(truncate=False)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Write file to silver in parquet

    # parquet_output_path = f"gs://earth_earthquake_data/Daily_data_pyspark/silver/{current_date}/PQ_flatten_data"
    # flattened_df.coalesce(1).write.parquet(parquet_output_path)

    # ---------------------------------------------------------------------------------------------------------------------------------

    # Read data from silver.parquet file
    parquet_file_path = (r"gs://earth_earthquake_data/Daily_data_pyspark/silver/20241123/PQ_flatten_data/part-00000"
                         r"-e61ae294-8d4e-4a88-bbbb-53409363b14b-c000.snappy.parquet")

    # Read the Parquet file
    pq_df = spark.read.parquet(parquet_file_path)
    df.show()

    # ---------------------------------------------------------------------------------------------------------------------------------

    pq_df2 = pq_df.withColumn('insert_date', current_timestamp())

    # Write data to bigquery

    pq_df2.write.format("bigquery") \
        .option("table", "earthquake-project-442611.earthquake_dataset.daily_data_table") \
        .option("writeMethod", "direct") \
        .save()
