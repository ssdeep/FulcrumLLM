package com.ssdeep.fulcrum.training

import com.ssdeep.fulcrum.transformer.GPTModel
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.generator.SimpleTextGenerator
import com.typesafe.config.ConfigFactory
import torch.noGrad

object TrainingSimpleTest extends App:
  val config = ConfigFactory.load()
  val gptConfig = GPTConfig.loadFromConfig(config)
  val gptModel = GPTModel(gptConfig)
  val simpleTextGenerator = SimpleTextGenerator(gptModel)
  val someInput = "Some Input"
  val output = simpleTextGenerator.generateSimpleText(TokenizerUtil.textToTokenIds(someInput), 5, gptConfig.contextLength)
  println(TokenizerUtil.tokenIdsToText(output))

  val inputTensors = torch.Tensor(
    Seq(
    TokenizerUtil.textToTokenIds("every effort moves").toSeq,
    TokenizerUtil.textToTokenIds("I really like").toSeq
    )
  )

  val targetTensors = torch.Tensor(
    Seq(
      TokenizerUtil.textToTokenIds(" effort moves you").toSeq,
      TokenizerUtil.textToTokenIds(" really like chocolate").toSeq
    )
  )

  println("Input tensors")
  println(inputTensors)
  val logits = noGrad {
    gptModel(inputTensors)
  }
  val probas = torch.softmax(logits, dim = -1, dtype = torch.float32)
  println("Probabilities")
  println(probas)
  val resultTokenIds = torch.argmax(probas, dim = -1, keepdim = true)
  println("Resulting token Ids")
  println(resultTokenIds)
  println("Target tensors")
  println(targetTensors)

  val targetProbas1 = probas.apply(0, Seq(0,1,2), targetTensors(0))
  val targetProbas2 = probas.apply(1, Seq(0, 1, 2), targetTensors(1))

  println(s"Prob?: $targetProbas1")
  println(s"Prob?: $targetProbas2")
  val logProbas = torch.log(torch.cat(Seq(targetProbas1, targetProbas2)))
  println(s"Log Probabilities: $logProbas")
  val avgLogProbas = torch.mean(logProbas)
  val negAvgLogProbas = -avgLogProbas

  println(s"Avg log probabilities: $avgLogProbas")
  println(s"Negative Avg Log Probs: $negAvgLogProbas")

  println(s"Logits shape: ${logits.shape}")
  println(s"Target shape: ${targetTensors.shape}")

  val logitsFlat = logits.flatten(0, 1)
  val targetFlat = targetTensors.flatten()

  println(s"Logits flattened ${logitsFlat.shape}")
  println(s"Targets flattened ${targetFlat.shape}")

  val loss = torch.nn.functional.crossEntropy(logitsFlat, targetFlat)
  println(loss)

