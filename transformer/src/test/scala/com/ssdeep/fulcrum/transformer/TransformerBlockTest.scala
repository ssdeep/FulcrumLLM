package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.typesafe.config.ConfigFactory
import org.bytedeco.pytorch.LongOptional
import torch.Tensor.fromNative

object TransformerBlockTest extends App:
  val config = ConfigFactory.load()
  val gptConfig = GPTConfig.loadFromConfig(config)
  val x = torch.rand(Seq(2, 4, gptConfig.embedDim), dtype = torch.float32)
  val block = TransformerBlock(gptConfig)
  val output = block(x)

  println(s"Input Shape: ${x.shape}")
  println(s"Output Shape: ${output.shape}")
  println(s"$output")
