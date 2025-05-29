package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.transformer.config.GPTConfig
import torch.{Float32, Int32, Int64, Tensor, nn}
import org.bytedeco.pytorch.global.torch as pytorch
import torch.nn.modules.TensorModule

case class MiniTransformerBlock(config: GPTConfig) extends TensorModule[Float32]:
  def apply(v1: Tensor[Float32]): Tensor[Float32] =
    v1

case class MiniLayerNorm(config: GPTConfig) extends TensorModule[Float32]:
  def apply(input: Tensor[Float32]): Tensor[Float32] =
    input

class MiniGPT(config: GPTConfig) extends torch.nn.Module:
  val tokenEmbed = new nn.Embedding(config.vocabSize.intValue(), config.embedDim)
  val posEmbed = new nn.Embedding(config.contextLength, config.embedDim)
  val dropEmbed = new nn.Dropout(config.dropRate)

  val trfBlocks = new nn.Sequential(
    (0 until config.numLayers).map(_ => MiniTransformerBlock(config)):_*
  )
  val finalNorm = MiniLayerNorm(config)
  val outHead = torch.nn.Linear(
    config.embedDim, config.vocabSize, addBias = false
  )

  def apply(input: Tensor[Int64]): Tensor[Float32] =
    val Seq(batchSize, contextLength) = input.shape
    val tokenEmbeds = tokenEmbed(input)
    val posEmbeds = posEmbed(
      torch.arange(end = contextLength, dtype = torch.int64)
    )
    var x = tokenEmbeds //+ posEmbeds
    x = dropEmbed(x)
    x = trfBlocks(x)
    x = finalNorm(x)
    val logits = outHead(x)
    logits

