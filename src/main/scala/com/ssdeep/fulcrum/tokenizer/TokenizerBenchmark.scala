package com.ssdeep.fulcrum.tokenizer

import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

object TokenizerBenchmark {
  def main(args: Array[String]): Unit = {
    val path = if (args.nonEmpty) args(0) else "/Users/dooby1/Downloads/wikiarchive/AllCombined.txt"
    val pbBuilder = new ProgressBarBuilder
    pbBuilder.setTaskName("Processing text lines")

    val texts = Files.readAllLines(Paths.get(path)).asScala.toList
    pbBuilder.setInitialMax(texts.size)
    val tokenizer =  Tokenizer.buildDefaulto200kBase

    val startTime = System.nanoTime()
    var totalTokens = 0
    var incorrectTokenization = 0

    for (text <- ProgressBar.wrap(texts.iterator.asJava, pbBuilder).asScala) {
      val tokens = tokenizer.encode(text)
      val decoded = tokenizer.decode(tokens)
      if (text != decoded) {
        incorrectTokenization = incorrectTokenization + 1
      }
      totalTokens += tokens.length
    }

    val endTime = System.nanoTime()
    val durationSeconds = (endTime - startTime) / 1e9

    println(f"Total tokens: $totalTokens")
    println(f"Time taken: $durationSeconds%.3f seconds")
    println(f"Throughput: ${totalTokens / durationSeconds}%.2f tokens/sec")
    println(s"Incorrect tokens counts: $incorrectTokenization")
  }
}
