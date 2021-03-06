package solution

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.{SparkConf, SparkContext}
import train.DigitRecognizer
import util.FileUtil

/**
  * Created by C.J.YOU on 2016/5/9.
  */
object MainDT {

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

    /*val splitData = data// .randomSplit(Array(0.9,0.1))
    // 使用交叉验证法训练数据和检验数据
    val trainData = data // splitData(0).cache()
    val testData = data // splitData(1).cache()

    val dtResult  = Seq(11,12,13,14,15,18,20).map { param =>
      val dtModel = DigitRecognizer.trainDTWithParams(trainData,param,Gini,10)
      val predictResult =  testData.map { labeledPoint =>
        val predicted = dtModel.predict(labeledPoint.features)
        (predicted,labeledPoint.label)
      }

      val metrics = new BinaryClassificationMetrics(predictResult)
      (param,metrics.areaUnderROC())
    }
    dtResult.foreach { case (param, acc) =>
      println(f"param:$param,auc:${acc * 100}")
    }*/

    // 预测数据 : 0.84486
   val dtModel = DigitRecognizer.trainDTWithParams(data,30,Gini,10)
    // predict test.csv filedata
    val test = sc.textFile(args(1)).map(_.split(","))
    val testLabels = test.map { lines =>
      val features = lines.map(_.toDouble)
      val label = dtModel.predict(Vectors.dense(features))
      label.toInt.toString
    }
    val arr = testLabels.zipWithIndex().map{x => (x._2 + 1 +",\""+x._1+"\"")}.collect()
    FileUtil.writeToFile(args(2),arr)
  }

}
