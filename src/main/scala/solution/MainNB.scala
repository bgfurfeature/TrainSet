package solution

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import train.DigitRecognizer
import util.FileUtil

/**
  * Created by C.J.YOU on 2016/5/9.
  */
object MainNB {

  def main(args: Array[String]) {

    val sc = new SparkContext(new SparkConf()
      .setAppName("DigitRe")
      .setMaster("local")
    )
    // checek data
    val allData = sc.textFile(args(0)).map(_.split(","))
    val data = allData.map{ lines =>
      val label = lines(0).toDouble
      val features = lines.slice(1,lines.length).map(_.toDouble)
      LabeledPoint(label,Vectors.dense(features))
    }

    val splitData = data.randomSplit(Array(0.8,0.2))
    // 使用交叉验证法训练数据和检验数据
    val trainData = splitData(0).cache()
    val testData = splitData(1).cache()


    val nbResult  = Seq(0.001,0.01,0.1,1.0,10.0).map { param =>
      val nbModel = DigitRecognizer.trainNBWithParams(trainData,param,"multinomial")
      val predictResult =  testData.map { labeledPoint =>
        val predicted = nbModel.predict(labeledPoint.features)
        (predicted,labeledPoint.label)
      }

      val metrics = new BinaryClassificationMetrics(predictResult)
      (param,metrics.areaUnderROC())
    }

    nbResult.foreach { case (param,acc) =>
      println(f"nb model with lambda:$param,modelTpye:multinomial,auc:${acc * 100}")
    }

    val nbBestModel = DigitRecognizer.trainNBWithParams(data,0.001,"multinomial")
    // predict test.csv filedata
    val test = sc.textFile(args(1)).map(_.split(","))
    val testLabels = test.map { lines =>
      val features = lines.map(_.toDouble)
      val label = nbBestModel.predict(Vectors.dense(features))
      label.toInt.toString
    }
    val arr = testLabels.zipWithIndex().map{x => (x._2 + 1 +",\""+x._1+"\"")}.collect()
    FileUtil.writeToFile(args(2),arr)
  }
}