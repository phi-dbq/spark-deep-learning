import pytest

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


@pytest.fixture(scope="class")
def spark_session(request):

    spark = SparkSession.builder.master("local[2]").appName("SparkDLPyTest").getOrCreate()
    sc = spark.sparkContext
    sqlCtx = SQLContext(sc, spark)
    request.cls.session = spark
    request.cls.sc = sc
    request.cls.sql = sqlCtx

    def finilizer():
        spark.stop()
        sc.stop()

    request.addfinalizer(finilizer)
