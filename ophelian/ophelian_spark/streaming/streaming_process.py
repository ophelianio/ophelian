from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext


class PySparkStreaming:
    def __init__(self, app_name, master_url):
        conf = SparkConf().setAppName(app_name).setMaster(master_url)
        self.sc = SparkContext(conf=conf)
        self.ssc = StreamingContext(self.sc, batchDuration=1)

    def create_dstream(self, hostname, port):
        self.dstream = self.ssc.socketTextStream(hostname, port)

    def process_dstream(self):
        self.processed_dstream = self.dstream.map(lambda x: x.upper())
        self.processed_dstream.count().pprint()

    def start(self):
        self.ssc.start()
        self.ssc.awaitTermination()

    def stop(self):
        self.ssc.stop(stopSparkContext=True, stopGraceFully=True)


if __name__ == "__main__":
    stream = PySparkStreaming("My Streaming App", "local[*]")
    stream.create_dstream("localhost", 9999)
    stream.process_dstream()
    stream.start()
    stream.stop()
