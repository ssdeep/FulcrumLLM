package com.ssdeep.fulcrum

import com.ssdeep.fulcrum.core.ClassNameLogger
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import torch.{Float32, Tensor}

import scala.io.*

object Main extends ClassNameLogger:
  def main(args: Array[String]): Unit =
    val tokenizer = Tokenizer.buildDefaultO200kBase
    val textFromVerdict = Source.fromResource("The_Verdict.txt")
      .getLines().mkString("")
    val MAX_LENGTH = 5L
    val BATCH_SIZE = 8
    val gptDataset = GPTDatasetV1(
        textFromVerdict,
        tokenizer = tokenizer,
        maxLength = MAX_LENGTH.intValue(),
        stride = 1,
        batchSize = BATCH_SIZE
    )

    torch.manualSeed(128L)

    val dataIter = gptDataset.dataLoader.iterator
    val (inputs, targets) = dataIter.next()
    val tokenEmbeddings: Tensor[Float32] = gptDataset ~> inputs // token embeddings
    val posEmbeddings = gptDataset.pos(torch.arange(end = MAX_LENGTH))

    logger.info(s"token embeddings shape: $tokenEmbeddings")
    logger.info(tokenizer.decode(inputs.toSeq.toList))
//    println(s"position embeddings: $posEmbeddings")
//    println("Token Embedding Layer")
//    println(gptDataset.tokenEmbeddingLayer)
//    println("Applying embedding")
//    println(tokenEmbeddings)
//    println("Applying token and positional embeddings to get input embeddings")
    val inputEmbeddings = tokenEmbeddings + posEmbeddings
//    println(s"Input Embeddding shape: ${inputEmbeddings.shape}")
//    println(s"Starting Input Embeddings: ${inputEmbeddings}")
    println("Done")


