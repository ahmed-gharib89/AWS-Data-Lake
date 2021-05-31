import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, monotonically_increasing_id


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID'] = config['KEYS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY'] = config['KEYS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Creates a spark session and configure it with needed packages"""

    spark = SparkSession.builder.config(
        "spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0").getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Loads song data into spark dataframe 
    then extract songs and artists tables and move them to s3 bucket

    Args:
        spark (spark session): spark session to use spark
        input_data (str): path to the input data
        output_data (str): path to the output data
    """
    # get filepath to song data file
    song_data = input_data + 'song_data/A/A/*/*.json'

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df['song_id', 'title', 'artist_id',
                     'year', 'duration'].dropDuplicates(['song_id'])

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').parquet(
        os.path.join(output_data, 'songs.parquet'), 'overwrite')

    # extract columns to create artists table
    artists_table = df['artist_id', 'artist_name', 'artist_location',
                       'artist_latitude', 'artist_longitude'].dropDuplicates(['artist_id'])

    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(
        output_data, 'artists.parquet'), 'overwrite')


def process_log_data(spark, input_data, output_data):
    """Loads log data into spark dataframe 
    then extract users, time and songplays tables and move them to s3 bucket

    Args:
        spark (spark session): spark session to use spark
        input_data (str): path to the input data
        output_data (str): path to the output data
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.filter(df.page == 'NextSong')

    # extract columns for users table
    users_table = df['userId', 'firstName', 'lastName',
                     'gender', 'level'].dropDuplicates(['userId'])

    # write users table to parquet files
    users_table.write.parquet(os.path.join(
        output_data, 'users.parquet'), 'overwrite')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'),
        month('datetime').alias('month'),
        year('datetime').alias('year')
    ).dropDuplicates(['start_time'])

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(
        os.path.join(output_data, 'time.parquet'), 'overwrite')

    # read in song data to use for songplays table
    song_df = spark.read.json(input_data + 'song_data/A/A/*/*.json')

    # extract columns from joined song and log datasets to create songplays table
    df = df.join(song_df, song_df.title == df.song)

    songplays_table = df.select(
        col('ts').alias('ts'),
        col('userId').alias('user_id'),
        col('level').alias('level'),
        col('song_id').alias('song_id'),
        col('artist_id').alias('artist_id'),
        col('ssessionId').alias('session_id'),
        col('location').alias('location'),
        col('userAgent').alias('user_agent'),
        col('year').alias('year'),
        month('datetime').alias('month')
    ).withColumn('songplay_id', monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(
        os.path.join(output_data, 'songplays.parquet'), 'overwrite')


def main():
    """Instiate spark session and call process song data then process log data
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://gharibudacity/"

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
