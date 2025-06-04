package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.typesafe.config.ConfigFactory

object GPTModelTest extends App:
  val tokenizer = Tokenizer.buildDefaultO200kBase
  val someRandomString = {
    torch.stack(
      Seq(
       torch.Tensor(tokenizer.encode("Some Random String for testing"))
      )
    )
  }
  val config = ConfigFactory.load()
  val gptConfig = GPTConfig.loadFromConfig(config)
  val gptModel = GPTModel(gptConfig)
  val output = gptModel(someRandomString)
  println(s"Total Params: ${gptModel.totalParams}")
  gptModel.printSizeInMB
  println(s"Input batch: $someRandomString")
  println(s"Output: $output")
