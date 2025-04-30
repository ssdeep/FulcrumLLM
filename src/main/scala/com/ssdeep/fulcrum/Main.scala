package com.ssdeep.fulcrum

import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import org.json4s.*
import org.json4s.JsonDSL.*
import org.json4s.native.JsonMethods
import org.json4s.native.JsonMethods.*

import java.io.{FileOutputStream, PrintWriter}
import scala.io.*

object Main:
  def main(args: Array[String]): Unit =
    val tokenizer = Tokenizer.buildDefaulto200kBase
    val textFromVerdict = Source.fromResource("The_Verdict.txt")
      .getLines().mkString("")
    val maxLength = 5L
    val gptDataset = GPTDatasetV1(
        textFromVerdict,
        tokenizer = tokenizer,
        maxLength = maxLength.intValue(),
        stride = 1
    )
    val tensorDataset = gptDataset.getDataset

    val dataLoader = torch.data.TupleDataLoader(
        tensorDataset,
        batchSize = 8,
        shuffle = false
    )
    println(dataLoader.head)
    torch.manualSeed(128L)
    val tokenEmbeddingLayer = new torch.nn.Embedding(tokenizer.tokenIdMap.size, 256)
    val dataIter = dataLoader.iterator
    val (inputs, targets) = dataIter.next()
    val tokenEmbeddings = tokenEmbeddingLayer.apply(inputs)
    val posEmbeddingLayer = new torch.nn.Embedding(maxLength.intValue(), 256)
    val posEmbeddings = posEmbeddingLayer.apply(torch.arange(end = maxLength))

    println(s"token embeddings shape: $tokenEmbeddings")
    println(s"position embeddings: $posEmbeddings")
    println("Token Embedding Layer")
    println(tokenEmbeddingLayer.weight)
    println("Applying embedding")
    println(tokenEmbeddingLayer.apply(torch.Tensor(Seq(1L,2L))))
    println("Applying token and positional embeddings to get input embeddings")
    val inputEmbeddings = tokenEmbeddings + posEmbeddings
    println(s"Input Embeddding shape: ${inputEmbeddings.shape}")
    println("Done")


