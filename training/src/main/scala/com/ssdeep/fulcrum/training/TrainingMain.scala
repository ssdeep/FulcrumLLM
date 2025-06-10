package com.ssdeep.fulcrum.training

import com.ssdeep.fulcrum.core.{ClassNameLogger, withProgressBuilder}
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import com.ssdeep.fulcrum.training.config.TrainingConfig
import com.ssdeep.fulcrum.transformer.GPTModel
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.generator.SimpleTextGenerator
import com.typesafe.config.ConfigFactory
import jdk.jshell.EvalException
import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}
import org.apache.pekko.actor.ActorSystem
import org.apache.pekko.stream.scaladsl.{Sink, Source}
import torch.data.TupleDataLoader
import torch.optim.Optimizer
import torch.{Device, Float32, Int64, Tensor}

import java.nio.file.{Files, Paths}
import scala.concurrent.{Await, Future}
import scala.jdk.CollectionConverters.*

object TrainingMain extends ClassNameLogger:

  implicit val system: ActorSystem = ActorSystem("TrainingMainSystem")
  implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global

  def calcLossPerBatch(inputBatch: Tensor[Int64], targetBatch: Tensor[Int64], model: GPTModel): (Tensor[Float32], Tensor[Float32]) = {
    val logits = model(inputBatch).flatten(0, 1)
    val loss = torch.nn.functional.crossEntropy(logits, targetBatch.flatten)
    (loss, logits)
  }

  def calcLossLoader(tupleDataLoader: Iterator[(Tensor[Int64], Tensor[Int64])], model: GPTModel,
                     limit: Option[Int] = None, count: Option[Int] = None): Float = {
    val iterableDataset = if limit.nonEmpty then tupleDataLoader.take(limit.get) else tupleDataLoader
    val lossLoaderSource = Source.fromIterator(
      () =>
      withProgressBuilder("Calculating Loss", iterableDataset, limit.orElse(count))
    )
      .mapAsync(4) {
        case (input, target) =>
          Future {
            val (lossByBatch, logits) = calcLossPerBatch(input, target, model) //.item
            val lossByBatchValue = lossByBatch.item
//            input.native.close()
//            target.native.close()
//            lossByBatch.native.close()
            lossByBatchValue
          }
      }
      .runWith(Sink.seq)
      .map(p => p.sum/p.length)
    Await.result(lossLoaderSource, scala.concurrent.duration.Duration.Inf)
  }

  def evaluateModel(model: GPTModel,
                    trainLoader: Iterator[(Tensor[Int64], Tensor[Int64])],
                    valLoader: Iterator[(Tensor[Int64], Tensor[Int64])],
                    evalIter: Option[Int] = None,
                    limit: Option[Int] = None
                   ): (Float, Float) = {
    model.eval()
      val trainLoss = torch.noGrad {
        calcLossLoader(trainLoader, model, evalIter)
      }
      logger.info(s"Train Loss after $evalIter iterations: $trainLoss")

      val valLoss = torch.noGrad {
        calcLossLoader(valLoader, model, evalIter)
      }
      logger.info(s"Validation Loss after $evalIter iterations: $valLoss")

    model.train()
    (trainLoss, valLoss)
  }

  def generateAndPrintSample(model: GPTModel, tokenizer: Tokenizer, startContext: String, simpleTextGenerator: SimpleTextGenerator) = {
    model.eval()
    val contextLength = model.cfg.contextLength
    val encoded = torch.Tensor(tokenizer.encode(startContext)).unsqueeze(0)
    val generatedTokens = simpleTextGenerator.generateSimpleText(
      encoded,
      50,
      50
    )
    val generatedText = tokenizer.decode(generatedTokens.toSeq.toList)
    logger.info(s"Generated Text: $generatedText")
    model.train()
  }

  def trainModelSimple(model: GPTModel,
                       tokenizer: Tokenizer,
                       trainLoader: Iterable[(Tensor[Int64], Tensor[Int64])],
                       trainCount: Int,
                       valLoader: Iterable[(Tensor[Int64], Tensor[Int64])],
                       valCount: Int,
                       optimizer: Optimizer,
                       numEpochs: Int,
                       evalFreq: Int, evalIter: Int
                      ): (List[Float], List[Float], List[Long]) = {
    val trainLosses = scala.collection.mutable.ArrayBuffer.empty[Float]
    val valLosses = scala.collection.mutable.ArrayBuffer.empty[Float]
    val trackTokensSeen = scala.collection.mutable.ArrayBuffer.empty[Long]
    val simpleTextGenerator = SimpleTextGenerator(model)
    var tokensSeen: Long = 0L
    var globalStep: Int = -1

    for (epoch <- withProgressBuilder("Epoch: ", (0 until numEpochs).iterator, Some(numEpochs))) {
      logger.info(s"Epoch Progress: $epoch")
      val losses = scala.collection.mutable.ArrayBuffer.empty[Tensor[Float32]]
      for ((inputBatch, targetBatch) <- withProgressBuilder(s"Training Progress for Epoch $epoch", trainLoader.iterator,
        Some(trainCount))) {
        globalStep += 1
        tokensSeen += inputBatch.numel
        trackTokensSeen.append(tokensSeen)
        optimizer.zeroGrad()
        val (loss, logits) = calcLossPerBatch(inputBatch, targetBatch, model)
        loss.backward()
        optimizer.step()
        trainLosses.append(loss.item)
        loss.native.close()
        logits.native.close()
        System.gc()
//        if globalStep % evalFreq == 0 && evalIter > 0 then
//          val (trainLoss, valLoss) = evaluateModel(model, trainLoader.iterator, valLoader.iterator, Some(evalIter))
//          trainLosses.append(trainLoss)
//          valLosses.append(valLoss)
      }
      generateAndPrintSample(model, tokenizer, "Functional Programming Description", simpleTextGenerator)
    }
    logger.info("Total Tokens Seen: {}", tokensSeen)
    logger.info("Losses: {}", trainLosses.mkString(", "))
    logger.info("Validation Losses: {}", valLosses.mkString(", "))
    logger.info("Tokens Seen: {}", trackTokensSeen.mkString(", "))
    (trainLosses.toList, valLosses.toList, trackTokensSeen.toList)
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
      batchSize = trainingConfig.batchSize,
      dimensionality = gptConfig.embedDim,
      shuffle = false
    ).dataLoaderAndCount
    val (valDataset, valCounts) = GPTDatasetV1(
      tokenIds = valData, tokenizer = tokenizer, maxLength = gptConfig.contextLength, stride = gptConfig.contextLength,
      batchSize = trainingConfig.batchSize,
      dimensionality = gptConfig.embedDim,
      shuffle = false
    ).dataLoaderAndCount
    val gptModel = GPTModel(gptConfig)

    logger.info("Initialized GPT Model with Total params: {}, expected size  {}", gptModel.totalParams, gptModel.printSizeInMB)
    val optimizer = torch.optim.AdamW(
      gptModel.parameters(true),
      lr = 0.0004, weightDecay = 0.1
    )
    val numEpochs = 1
    val evalFreq = 2
    val evalIter = 5
    val trainingCount = trainingConfig.limit.getOrElse(trainCounts)
    val valCount = trainingConfig.limit.getOrElse(valCounts)
    val loss = trainModelSimple(gptModel, tokenizer,
      trainDataset.take(trainingCount), trainingCount,
      valDataset.take(valCount),
      valCount,
      optimizer,
      numEpochs,
      evalFreq,
      evalIter)

    println(s"Total Loss: $loss")
