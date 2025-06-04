package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.generator.SimpleTextGenerator
import com.typesafe.config.ConfigFactory

object SimpleTextGeneratorTest extends App:
  val tokenizer = Tokenizer.buildDefaultO200kBase
  val someRandomString = {
    torch.stack(
      Seq(
        torch.Tensor(tokenizer.encode("Some Random String for testing")),
        torch.Tensor(tokenizer.encode("More Random String for testing"))
      )
    )
  }
  val config = ConfigFactory.load()
  val gptConfig = GPTConfig.loadFromConfig(config)
  val gptModel = GPTModel(gptConfig)

  val simpleTextGenerator = SimpleTextGenerator(gptModel)
  val nextString = simpleTextGenerator.generateSimpleText(someRandomString, 6, 20)
  println(s"Output $nextString")
  println(s"Output as sequence ${nextString.toSeq}")
  println(s"Output squeezed ${nextString.squeeze(0)}")
  val decoded = tokenizer.decode(nextString.toSeq.toList)
  println(s"Decoded Strings: $decoded")
