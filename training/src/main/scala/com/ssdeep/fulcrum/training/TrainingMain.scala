package com.ssdeep.fulcrum.training

import com.ssdeep.fulcrum.core.{ClassNameLogger, withProgressBuilder}
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import com.ssdeep.fulcrum.training.config.TrainingConfig
import com.ssdeep.fulcrum.transformer.GPTModel
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.typesafe.config.ConfigFactory
import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}
import torch.data.TupleDataLoader
import torch.{Device, Int64, Tensor}

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

object TrainingMain extends ClassNameLogger:

  def calcLossPerBatch(inputBatch: Tensor[Int64], targetBatch: Tensor[Int64], model: GPTModel) = {
    val logits = model(inputBatch).flatten(0, 1)
    val loss = torch.nn.functional.crossEntropy(logits, targetBatch.flatten)
    logits.native.deallocate()
    loss
  }

  def calcLossLoader(tupleDataLoader: Iterator[(Tensor[Int64], Tensor[Int64])], model: GPTModel,
                     limit: Option[Int] = None, count: Option[Int] = None)
  = {
    val iterableDataset = if limit.nonEmpty then tupleDataLoader.take(limit.get) else tupleDataLoader

    val (totalLoss, totalCount) = withProgressBuilder("Calculating Loss", iterableDataset, limit.orElse(count))
      .foldLeft(0.0, 0) {
      case ((loss, count), (input, target)) =>
        val lossByBatch = calcLossPerBatch(input, target, model)//.item
        val lossByBatchValue = lossByBatch.item
        input.native.deallocate()
        target.native.deallocate()
        lossByBatch.native.deallocate(true)
        (loss + lossByBatchValue, count + 1)
    }
    totalLoss/totalCount
  }

  def main(args: Array[String]): Unit =
    val config = ConfigFactory.load()
    val gptConfig = GPTConfig.loadFromConfig(config)
    val trainingConfig = TrainingConfig.loadFromConfig(config)

    val pbBuilder = new ProgressBarBuilder
    pbBuilder.setTaskName("Encoding text lines")

    val textCorpus = Files.readAllLines(Paths.get(trainingConfig.file))
      .asScala.toList//.mkString("\n")
    logger.info(s"Total Characters: ${textCorpus.map(_.length).sum}")
//    logger.info(s"Number of tokens: ${tokenizedCorpus.length}")
//    pbBuilder.setInitialMax(texts.size)
    val tokenizer = Tokenizer.buildDefaultO200kBase
    val tokenizedCorpus = ProgressBar.wrap(textCorpus.asJava, pbBuilder).asScala.flatMap {
      text =>
        tokenizer.encode(text)
    }.toList
    logger.info(s"Total Tokens: ${tokenizedCorpus.length}")
    val trainSplit = (tokenizedCorpus.length * trainingConfig.ratio).intValue()
    val (trainData, valData) = tokenizedCorpus.splitAt(trainSplit)
    val (trainDataset, trainCounts) = GPTDatasetV1(
      tokenIds = trainData, tokenizer = tokenizer, maxLength = gptConfig.contextLength, stride = gptConfig.contextLength,
      batchSize = 2,
      dimensionality = gptConfig.embedDim,
      shuffle = false
    ).dataLoaderAndCount
    val (valDataset, valCounts) = GPTDatasetV1(
      tokenIds = valData, tokenizer = tokenizer, maxLength = gptConfig.contextLength, stride = gptConfig.contextLength,
      batchSize = 2,
      dimensionality = gptConfig.embedDim,
      shuffle = false
    ).dataLoaderAndCount
    val gptModel = GPTModel(gptConfig)
    val loss = calcLossLoader(trainDataset, gptModel, limit = trainingConfig.limit, Some(trainCounts))
    println(s"Total Loss: $loss")
