import os
from pyspark.sql import SparkSession, functions as F, types as T

def main():
    # Paths as described in assessment
    input_path = '/home/prac/test3/input/HotelReservations.csv'
    output_path = '/home/prac/test3/output/result.txt'

    # Start Spark session
    spark = SparkSession.builder.appName('HotelReservationAssessment').getOrCreate()

    # Define schema with appropriate column names and types
    schema = T.StructType([
        T.StructField('Booking_ID', T.StringType(), True),
        T.StructField('Num_Adults', T.IntegerType(), True),
        T.StructField('Num_Children', T.IntegerType(), True),
        T.StructField('Num_Weekend_Nights', T.IntegerType(), True),
        T.StructField('Num_Week_Nights', T.IntegerType(), True),
        T.StructField('Meal_Plan', T.StringType(), True),
        T.StructField('Car_Parking', T.IntegerType(), True),
        T.StructField('Room_Type', T.StringType(), True),
        T.StructField('Lead_Time', T.IntegerType(), True),
        T.StructField('Arrival_Year', T.IntegerType(), True),
        T.StructField('Arrival_Month', T.StringType(), True),
        T.StructField('Arrival_Date', T.IntegerType(), True),
        T.StructField('Segment_Type', T.StringType(), True),
        T.StructField('Repeated_Guest', T.IntegerType(), True),
        T.StructField('Prev_Cancellations', T.IntegerType(), True),
        T.StructField('Prev_Bookings_Not_Cancelled', T.IntegerType(), True),
        T.StructField('Avg_Room_Price', T.DoubleType(), True),
        T.StructField('Special_Requests', T.IntegerType(), True),
        T.StructField('Booking_Status', T.StringType(), True)
    ])

    # Load CSV file
    df = spark.read.csv(input_path, header=True, schema=schema)

    # Question 2: proportion of bookings by market segment type
    total_count = df.count()
    proportions = (
        df.groupBy('Segment_Type')
          .count()
          .withColumn('percentage', F.col('count') * 100.0 / total_count)
          .orderBy('Segment_Type')
    )

    # Question 3: month with highest average lead time
    avg_lead_by_month = (
        df.groupBy('Arrival_Month')
          .agg(F.avg('Lead_Time').alias('avg_lead_time'))
    )
    max_month_row = avg_lead_by_month.orderBy(F.desc('avg_lead_time')).first()

    # Question 4: average room price per night for each room type
    nights_col = F.col('Num_Week_Nights') + F.col('Num_Weekend_Nights')
    price_col = nights_col * F.col('Avg_Room_Price')
    avg_price_room = (
        df.withColumn('total_nights', nights_col)
          .withColumn('total_spending', price_col)
          .groupBy('Room_Type')
          .agg(
              F.sum('total_spending').alias('sum_spending'),
              F.sum('total_nights').alias('sum_nights')
          )
          .withColumn('avg_price_per_night', F.col('sum_spending') / F.col('sum_nights'))
          .select('Room_Type', 'avg_price_per_night')
          .orderBy('Room_Type')
    )

    # Question 5: longest booking among customers with children in 2018
    longest_booking_row = (
        df.filter( (F.col('Arrival_Year') == 2018) & (F.col('Num_Children') > 0) )
          .withColumn('total_nights', nights_col)
          .agg(F.max('total_nights').alias('max_nights'))
          .first()
    )

    # Prepare output lines
    lines = []

    lines.append('Q2: Proportion of bookings by market segment type')
    for row in proportions.collect():
        lines.append(f"{row['Segment_Type']}: {row['percentage']}")

    lines.append('Q3: Month with highest average lead time')
    if max_month_row:
        lines.append(f"{max_month_row['Arrival_Month']}: {max_month_row['avg_lead_time']}")
    else:
        lines.append('No data')

    lines.append('Q4: Average room price per night by room type')
    for row in avg_price_room.collect():
        lines.append(f"{row['Room_Type']}: {row['avg_price_per_night']}")

    lines.append('Q5: Longest booking with children in 2018')
    if longest_booking_row:
        lines.append(str(longest_booking_row['max_nights']))
    else:
        lines.append('No data')

    # Write to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(str(line) + '\n')

    spark.stop()

if __name__ == '__main__':
    main()
